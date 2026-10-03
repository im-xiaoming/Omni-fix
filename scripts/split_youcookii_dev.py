#!/usr/bin/env python3
"""
split_youcookii_dev.py
======================
Hold out a dev split from the YouCookII *train* manifest, by video.

YouCookII's test annotations are not public, so val is used as the test set and
checkpoints have to be chosen on a dev split carved out of train. The split is
made per video -- every event of a video lands on the same side -- so dev never
shares a video with what the model was fine-tuned on. It is stratified by
``recipe_type`` (read from ``youcookii_train_preprocess.json``) so dev covers the
same dishes as train.

Inputs
------
    --manifest      train_omni_video.jsonl          (from convert_youcookii.py)
    --preprocess    youcookii_train_preprocess.json (for recipe_type)

Outputs (next to --manifest unless --out-dir is given)
-------
    train_omni_video.trainsplit.jsonl   fine-tune Omni on this
    train_omni_video.dev.jsonl          choose checkpoints on this
    dev_videos.txt                      one video id per line; give the same list
                                        to UniAV so both backbones hold out the
                                        same videos

The split depends only on --seed and --dev-ratio, so rerunning reproduces it.

Usage
-----
    python scripts/split_youcookii_dev.py \\
        --manifest   "D:/Học/KL/Data/YouCookII/metadata/train_omni_video.jsonl" \\
        --preprocess "D:/Học/KL/Data/YouCookII/metadata/youcookii_train_preprocess.json"
"""

import argparse
import json
import os
import random
from collections import defaultdict


def video_id(record):
    return os.path.splitext(os.path.basename(record["video"]))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preprocess", required=True,
                        help="youcookii_train_preprocess.json, read for recipe_type")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    with open(args.preprocess, encoding="utf-8") as f:
        database = json.load(f)["database"]
    recipe_of = {ev["video_id"]: str(ev.get("recipe_type", "unknown")) for ev in database.values()}

    videos_by_recipe = defaultdict(set)
    for r in records:
        vid = video_id(r)
        videos_by_recipe[recipe_of.get(vid, "unknown")].add(vid)

    # At least one dev video per recipe that has two or more, so no dish is
    # missing from dev; a recipe with a single video stays in train.
    rng = random.Random(args.seed)
    dev_videos = set()
    for recipe in sorted(videos_by_recipe):
        vids = sorted(videos_by_recipe[recipe])
        if len(vids) < 2:
            continue
        rng.shuffle(vids)
        dev_videos.update(vids[:max(1, round(len(vids) * args.dev_ratio))])

    train_part = [r for r in records if video_id(r) not in dev_videos]
    dev_part = [r for r in records if video_id(r) in dev_videos]

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.manifest))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.manifest))[0]
    paths = {
        "train": os.path.join(out_dir, f"{stem}.trainsplit.jsonl"),
        "dev": os.path.join(out_dir, f"{stem}.dev.jsonl"),
        "videos": os.path.join(out_dir, "dev_videos.txt"),
    }
    for key, rows in (("train", train_part), ("dev", dev_part)):
        with open(paths[key], "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(paths["videos"], "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sorted(dev_videos)) + "\n")

    n_videos = len({video_id(r) for r in records})
    print(f"recipes: {len(videos_by_recipe)} | videos: {n_videos} | events: {len(records)}")
    print(f"train : {n_videos - len(dev_videos):5d} videos {len(train_part):6d} events -> {paths['train']}")
    print(f"dev   : {len(dev_videos):5d} videos {len(dev_part):6d} events -> {paths['dev']}")
    print(f"dev video ids -> {paths['videos']}")
    assert not {video_id(r) for r in train_part} & dev_videos
    assert len(train_part) + len(dev_part) == len(records)


if __name__ == "__main__":
    main()
