"""Command-line interface entry points.

Two sub-commands are exposed::

    python -m omniretriever.cli extract  [args ...]
    python -m omniretriever.cli evaluate [args ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger("omniretriever")


def extract_main(argv: list[str] | None = None) -> int:
    """Entry point for ``omniretriever-extract``.

    The manifest is a JSON or JSONL file with one record per line/element::

        {"id": "...", "text": "...", "video": "videos/clip.mp4", "audio": "videos/clip.wav"}

    Any modality field may be missing. Every modality a record can support is
    extracted in the same run, so one pass over the canonical manifest yields
    ``text``, ``video``, ``audio``, ``av``, ``tv`` and ``at`` keys as the fields
    allow -- enough to score all twelve benchmark directions. The ``av``
    embedding needs both ``video`` and ``audio`` to be present, and is read from
    the video container, whose audio track is decoded alongside the frames;
    ``tv`` and ``at`` pair a media stream with the caption.

    Records are encoded in batches, and the partial result is checkpointed to
    ``OUTPUT.partial.npz`` so an interrupted run resumes where it stopped
    instead of starting over.
    """
    parser = _build_extract_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.stem + ".partial.npz")

    records = _load_manifest(args.manifest)
    out: dict[str, np.ndarray] = {}
    if args.resume and partial_path.is_file():
        with np.load(partial_path) as blob:
            out = {key: blob[key] for key in blob.files}
        logger.info("Resuming from %s (%d embeddings already done)", partial_path, len(out))

    todo = _plan(records, out)
    if not todo:
        logger.info("Nothing left to extract; writing %d embeddings", len(out))
        return _finalise(out, output_path, partial_path)

    from omniretriever import OmniRetriever

    model = OmniRetriever.from_pretrained(
        base_model=args.base_model,
        adapter=args.adapter,
        device=args.device,
        dtype=args.dtype,
        duplicate_audio_tokens=args.duplicate_audio_tokens,
    )
    # tv and at take two arguments, so their batches arrive as (media, caption)
    # pairs and are transposed back into two parallel lists here.
    encoders = {
        "text": model.encode_text,
        "video": model.encode_video,
        "audio": model.encode_audio,
        "av": model.encode_av,
        "tv": lambda values: model.encode_tv(*_unzip(values)),
        "at": lambda values: model.encode_at(*_unzip(values)),
    }

    batches = [
        (modality, items[i:i + args.batch_size])
        for modality, items in todo
        for i in range(0, len(items), args.batch_size)
    ]
    for done, (modality, chunk) in enumerate(_progress(batches, "extracting"), start=1):
        vectors = encoders[modality]([value for _, value in chunk]).cpu().float().numpy()
        for (record_id, _), vector in zip(chunk, vectors):
            out[f"{record_id}__{modality}"] = vector
        if done % args.checkpoint_every == 0:
            _checkpoint(out, partial_path)

    return _finalise(out, output_path, partial_path)


def evaluate_main(argv: list[str] | None = None) -> int:
    """Entry point for ``omniretriever-evaluate``."""
    parser = _build_evaluate_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from omniretriever.evaluation.score import score_benchmark, score_directory

    src = Path(args.embeddings)
    if src.is_dir():
        results = score_directory(src)
    else:
        results = {src.stem: score_benchmark(src)}

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _plan(records: list[dict], done: Mapping[str, np.ndarray]) -> list[tuple[str, list]]:
    """Group the outstanding work by modality.

    Returns one ``(modality, [(record_id, encoder_input), ...])`` entry per
    modality, skipping anything already present in ``done``. Grouping this way
    keeps every batch homogeneous, which is what the ``encode_*`` helpers need:
    each builds a single prompt template for the whole batch.
    """
    sources = {
        "text": lambda r: r.get("text"),
        "video": lambda r: r.get("video"),
        "audio": lambda r: r.get("audio"),
        # Both streams are decoded from the video container, so the audio field
        # only gates the branch; see OmniRetriever.encode_av.
        "av": lambda r: r.get("video") if r.get("audio") else None,
        # tv and at ride on encoders added downstream (see the note in
        # omniretriever.inference.encode); their inputs are (media, caption)
        # pairs rather than a single path.
        "tv": lambda r: (r["video"], r["text"]) if r.get("video") and r.get("text") else None,
        "at": lambda r: (r["audio"], r["text"]) if r.get("audio") and r.get("text") else None,
    }

    plan: list[tuple[str, list]] = []
    for modality, source in sources.items():
        items = [
            (record["id"], value)
            for record in records
            for value in (source(record),)
            if value is not None and f"{record['id']}__{modality}" not in done
        ]
        if items:
            plan.append((modality, items))
    return plan


def _unzip(pairs: Sequence[tuple]) -> tuple[list, list]:
    """Turn ``[(a1, b1), (a2, b2), ...]`` into ``([a1, a2, ...], [b1, b2, ...])``."""
    return [a for a, _ in pairs], [b for _, b in pairs]


def _progress(items: Sequence, description: str) -> Iterable:
    """Wrap ``items`` in a tqdm bar, falling back to a plain iterator."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        logger.info("%s: %d batches (install tqdm for a progress bar)", description, len(items))
        return items
    return tqdm(items, desc=description, unit="batch")


def _checkpoint(out: Mapping[str, np.ndarray], partial_path: Path) -> None:
    """Write the partial result, replacing the previous checkpoint atomically."""
    tmp_path = partial_path.with_name(partial_path.stem + ".tmp.npz")
    np.savez(tmp_path, **out)
    os.replace(tmp_path, partial_path)


def _finalise(out: Mapping[str, np.ndarray], output_path: Path, partial_path: Path) -> int:
    np.savez(output_path, **out)
    partial_path.unlink(missing_ok=True)
    logger.info("Wrote %d embeddings to %s", len(out), output_path)
    return 0


def _build_extract_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omniretriever-extract",
        description="Extract OmniRetriever-7B embeddings for a manifest of records.",
    )
    parser.add_argument("manifest", help="Path to a JSON / JSONL manifest.")
    parser.add_argument("--base-model", required=True, help="Path to the WAVE-7B backbone.")
    parser.add_argument("--adapter", required=True, help="Path to the LoRA adapter directory.")
    parser.add_argument("--output", required=True, help="Output .npz file.")
    parser.add_argument("--device", default="cuda", help="Torch device (default cuda).")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float32", "bfloat16", "float16"),
        help="Inference precision (default bfloat16).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Records encoded per forward pass (default 1). Raising it above 1 changes the "
            "embeddings for every sample that is not the longest in its batch: the fusion head "
            "pools a fixed final position (hidden_states[:, -1, :]) and the video/audio branches "
            "do not compensate for the padding that batching introduces. Text-only manifests are "
            "unaffected because the processor left-pads the text stream."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write the partial .npz every N batches (default 25).",
    )
    parser.add_argument(
        "--no-duplicate-audio-tokens",
        dest="duplicate_audio_tokens",
        action="store_false",
        help=(
            "Reproduce the released audio behaviour, which hands the model half the audio "
            "token slots the interleaved BEATs branch fills, so half the feature block is "
            "silently dropped. Use it only to A/B against the corrected path."
        ),
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore an existing OUTPUT.partial.npz and extract everything again.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def _build_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omniretriever-evaluate",
        description="Score paired embeddings against the OmniRetriever-Bench setup.",
    )
    parser.add_argument(
        "embeddings",
        help="Either a single .bin / .npz file or a directory of them.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def _load_manifest(path: str) -> list[dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    blob = json.loads(text)
    if isinstance(blob, list):
        return blob
    raise ValueError(f"Manifest must be a list (.json) or JSONL; got {type(blob).__name__}.")


def _setup_logging(verbose: int) -> None:
    level = logging.WARNING - 10 * min(verbose, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    """Dispatch to ``extract`` or ``evaluate`` based on the first argument."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: python -m omniretriever.cli {extract,evaluate} [args ...]",
            file=sys.stderr,
        )
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "extract":
        return extract_main(rest)
    if cmd == "evaluate":
        return evaluate_main(rest)
    print(f"unknown sub-command {cmd!r}; expected 'extract' or 'evaluate'.", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
