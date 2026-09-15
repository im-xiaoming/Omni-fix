#!/usr/bin/env python3
"""Report how many manifest records can actually feed TupleInfoNCE.

Run this before training. ``obj1`` is only non-zero when *every* sample in an
optimiser step carries augmented video, audio and a caption, because the
collator drops a field for the whole batch as soon as one sample lacks it
(``data_qwen.py::DataCollatorForOmniDataset``). A modest per-record gap
therefore turns into a near-total obj1 blackout:

    P(obj1 fires) = coverage ** (batch_size * grad_accum)

So 79% coverage with batch 4 x accum 4 leaves obj1 alive in 2% of steps -- which
is what a full YouCookII run actually produced before the audio clipping fix.

This checks the two things that fail silently at load time:

* the audio file decodes to a non-empty waveform *after* ``timestamps`` clipping
  (a per-segment .wav clipped by full-video timestamps comes back empty), and
* the record has a video, an audio field and a caption at all.

Usage::

    VIDEO_ROOT=/content/data/YouCookII/videos \\
    AUDIO_ROOT=/content/data/YouCookII/audio \\
    python scripts/check_tuple_data.py /content/data/YouCookII/metadata/train_omni.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SAMPLE_RATE = 16_000


def resolve(filename: str, env_var: str) -> str:
    """Mirror of ``data_qwen.resolve_media_path``."""
    if not filename or os.path.isabs(filename):
        return filename
    root = os.environ.get(env_var, "")
    return os.path.join(root, filename) if root else filename


def caption_of(record: dict) -> str:
    """The ``gpt`` turn, which is what becomes ``label`` in the loader."""
    for turn in record.get("conversations", []):
        if turn.get("from") == "gpt":
            return turn.get("value", "")
    return ""


def audio_samples_after_clip(path: str, timestamps) -> int | None:
    """Samples left after the loader's clipping, or None when unreadable."""
    import soundfile as sf

    try:
        info = sf.info(path)
    except Exception:
        return None

    total = int(info.frames * SAMPLE_RATE / info.samplerate)
    if timestamps is None:
        return total

    start = int(float(timestamps[0]) * SAMPLE_RATE)
    end = int(float(timestamps[1]) * SAMPLE_RATE)
    # Matches the fixed loader: clip only when the window starts inside the file.
    if start >= total:
        return total
    return max(0, min(end, total) - start)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="check only the first N records (0 = all)")
    ap.add_argument("--workers", type=int, default=16,
                    help="threads for the audio header reads (default 16). Reading "
                         "thousands of files off a Google Drive FUSE mount is latency "
                         "bound, so threads help a lot there.")
    args = ap.parse_args(argv)

    print(f"manifest   : {args.manifest}", flush=True)
    print(f"AUDIO_ROOT : {os.environ.get('AUDIO_ROOT', '(chua dat)')}", flush=True)
    print(f"VIDEO_ROOT : {os.environ.get('VIDEO_ROOT', '(chua dat)')}", flush=True)

    lines = Path(args.manifest).read_text(encoding="utf-8").splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    if args.limit:
        records = records[:args.limit]
    print(f"record     : {len(records):,}\n", flush=True)

    stats = {"no_video": 0, "no_audio_field": 0, "no_caption": 0,
             "audio_missing": 0, "audio_empty": 0, "ok": 0}
    examples: list[str] = []

    # Cheap checks first, so the slow per-file reads only cover what needs them.
    need_audio = []
    for rec in records:
        if not rec.get("video"):
            stats["no_video"] += 1
        elif not caption_of(rec):
            stats["no_caption"] += 1
        elif not rec.get("audio"):
            stats["no_audio_field"] += 1
        else:
            need_audio.append(rec)

    print(f"doc header cua {len(need_audio):,} file audio "
          f"({args.workers} luong)...", flush=True)

    from concurrent.futures import ThreadPoolExecutor

    def check(rec):
        path = resolve(rec["audio"], "AUDIO_ROOT")
        return rec, path, audio_samples_after_clip(path, rec.get("timestamps"))

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for rec, path, n in pool.map(check, need_audio):
            done += 1
            if done % 250 == 0 or done == len(need_audio):
                print(f"  {done:,}/{len(need_audio):,}  ok={stats['ok']:,} "
                      f"rong={stats['audio_empty']:,} thieu={stats['audio_missing']:,}",
                      flush=True)
            if n is None:
                stats["audio_missing"] += 1
                if len(examples) < 5:
                    examples.append(f"  khong doc duoc: {path}")
            elif n < SAMPLE_RATE:
                stats["audio_empty"] += 1
                if len(examples) < 5:
                    examples.append(
                        f"  rong/qua ngan ({n} mau): {rec.get('id')} ts={rec.get('timestamps')}")
            else:
                stats["ok"] += 1

    total = len(records)
    cov = stats["ok"] / total if total else 0.0
    per_step = args.batch_size * args.grad_accum

    print("=" * 64)
    print(f"  {total:,} record trong {args.manifest}")
    print("=" * 64)
    for k, v in stats.items():
        if v:
            print(f"  {k:16s} {v:7,d}  ({100.0*v/total:5.1f}%)")
    print("-" * 64)
    print(f"  coverage         {cov:7.1%}")
    print(f"  batch {args.batch_size} x accum {args.grad_accum} = {per_step} sample/step")
    print(f"  => P(obj1 khac 0) ~ {cov ** per_step:.1%}")
    print("=" * 64)
    if examples:
        print("vi du:")
        print("\n".join(examples))

    if cov < 0.95:
        print("\nCANH BAO: coverage < 95%. obj1 se im lang gan nhu ca run.")
        return 1
    print("\nOK — TupleInfoNCE se chay that.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
