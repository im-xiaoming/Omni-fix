#!/usr/bin/env python3
"""Score an embeddings ``.npz`` produced by ``omniretriever.cli extract``.

    python scripts/score_embeddings.py embeddings.npz
    python scripts/score_embeddings.py embeddings.npz --output metrics.json

The file holds one vector per ``{record_id}__{modality}`` key. Every direction
in ``omniretriever.cli.DIRECTIONS`` whose two sides are both present gets
scored, so a run restricted with ``--modalities`` simply yields fewer rows.

The built-in ``omniretriever.cli evaluate`` does not read this layout: it wants
a ``.bin`` holding ``text_embeds`` / ``mllm_embeds`` and scores only ``t2m`` and
``m2t``.

Each direction is scored over the records that carry *both* of its modalities.
Recall depends on gallery size, so the per-direction count is printed alongside
the metrics -- two directions are only comparable when their counts match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from omniretriever.cli import DIRECTIONS  # noqa: E402
from omniretriever.evaluation.metrics import all_metrics  # noqa: E402
from omniretriever.evaluation.score import cosine_similarity  # noqa: E402

SINGLE = ("t2v", "v2t", "t2a", "a2t", "v2a", "a2v")
DUAL = ("t2av", "av2t", "a2tv", "tv2a", "v2at", "at2v")


def load_by_modality(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Group the flat ``{id}__{modality}`` keys into ``{modality: {id: vector}}``."""
    grouped: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path) as blob:
        for key in blob.files:
            record_id, _, modality = key.rpartition("__")
            grouped.setdefault(modality, {})[record_id] = blob[key]
    return grouped


def score(path: Path) -> dict[str, dict[str, float]]:
    grouped = load_by_modality(path)
    results: dict[str, dict[str, float]] = {}

    for name, (query, gallery) in DIRECTIONS.items():
        if query not in grouped or gallery not in grouped:
            continue
        ids = sorted(set(grouped[query]) & set(grouped[gallery]))
        if not ids:
            continue
        q = np.stack([grouped[query][i] for i in ids]).astype(np.float64)
        g = np.stack([grouped[gallery][i] for i in ids]).astype(np.float64)
        metrics = all_metrics(cosine_similarity(q, g))
        metrics["gallery"] = len(ids)
        results[name] = metrics

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("embeddings", help="the .npz written by `cli extract`")
    parser.add_argument("--output", default=None, help="also write the metrics as JSON")
    args = parser.parse_args(argv)

    results = score(Path(args.embeddings))
    if not results:
        print("No direction could be scored: no record carries both sides of any pair.")
        return 1

    print(f"{'huong':<7}{'R@1':>9}{'R@5':>9}{'R@10':>9}{'MRR':>9}{'gallery':>9}")
    for name, m in results.items():
        print(f"{name:<7}{m['R@1']:>9.4f}{m['R@5']:>9.4f}{m['R@10']:>9.4f}"
              f"{m['MRR']:>9.4f}{m['gallery']:>9d}")

    print()
    for label, group in (("AVG-single", SINGLE), ("AVG-dual", DUAL), ("AVG-all", tuple(results))):
        scored = [results[d]["R@1"] for d in group if d in results]
        if scored:
            print(f"{label:<11} R@1: {np.mean(scored):.4f}   ({len(scored)} huong)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nDa ghi metrics vao {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
