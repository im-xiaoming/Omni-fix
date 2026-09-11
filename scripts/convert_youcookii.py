#!/usr/bin/env python3
"""
convert_youcookii.py
====================
Convert the raw YouCookII metadata JSONL files into the ``train_omni.jsonl``
format expected by ``data_qwen.py`` / ``LazySupervisedDataset``.

Key things this script does
---------------------------
1. Reads the original ``youcookii_train.jsonl`` (and optionally ``youcookii_val.jsonl``).
2. Each source record is **already one cooking event** (segment).  The ``timestamps``
   field tells the loader which slice of the full video / audio to use -- no further
   splitting is needed.
3. Normalises the record to the exact schema expected by the trainer.
4. Writes ``train_omni.jsonl`` (and ``val_omni.jsonl``) that can be passed directly
   to ``DATA_PATH=...train_omni.jsonl bash training/train.sh``.

Usage
-----
    python scripts/convert_youcookii.py \\
        --train  "D:/Hoc/KL/Data/YouCookII/YouCookII/metadata/youcookii_train.jsonl" \\
        --val    "D:/Hoc/KL/Data/YouCookII/YouCookII/metadata/youcookii_val.jsonl"   \\
        --out_dir "D:/Hoc/KL/Data/YouCookII/YouCookII/metadata"

Then launch training with:

    VIDEO_ROOT="D:/Hoc/KL/Data/YouCookII/YouCookII/videos" \\
    AUDIO_ROOT="D:/Hoc/KL/Data/YouCookII/YouCookII/audio"  \\
    WAVE_PATH="D:/Hoc/KL/Code/Omni/WAVE_HOME/WAVE-7B"      \\
    BEATS_PATH="D:/Hoc/KL/Code/Omni/WAVE_HOME/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" \\
    DATA_PATH="D:/Hoc/KL/Data/YouCookII/YouCookII/metadata/train_omni.jsonl" \\
    OUTPUT_DIR="D:/Hoc/KL/Code/Omni/output/omniretriever_youcook" \\
    bash training/train.sh

Note on path resolution
-----------------------
Filenames in the output are kept as **bare basenames** (e.g. ``GLd3aX16zBg.mp4``).
The data loader in ``data_qwen.py`` automatically prepends ``$VIDEO_ROOT`` /
``$AUDIO_ROOT`` to any non-absolute path, so you only ever need to change those
two environment variables -- not the JSONL itself.

Note on event segmentation
--------------------------
Each line in youcookii_train.jsonl is already ONE cooking step (event), with
``timestamps: [start, end]``. The data loader in ``data_qwen.py`` calls
``video_decord(video_file, timestamps=timestamps)`` which decodes only the
requested clip from the full video file. No pre-splitting of videos is needed.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def build_omni_record(src: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one raw YouCookII record to the omni training schema.

    Source schema (youcookii_train.jsonl)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      "id": "GLd3aX16zBg_1",
      "conversations": [
        {"from": "human", "value": "<video>\\nPlease describe the video."},
        {"from": "gpt",   "value": "place a slice of cheese on the bread"}
      ],
      "video":      "GLd3aX16zBg.mp4",    <- bare filename
      "audio":      "GLd3aX16zBg_1.wav",  <- bare filename
      "timestamps": [114.0, 127.0],       <- already one event/segment
      "text":       "place a slice of cheese on the bread"
    }

    Target schema (train_omni.jsonl)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      "id": "GLd3aX16zBg_1",
      "type": "retrieval",               <- required by LazySupervisedDataset
      "conversations": [...],            <- human / gpt turns, normalised
      "video":      "GLd3aX16zBg.mp4",  <- bare filename; VIDEO_ROOT prepended at load time
      "audio":      "GLd3aX16zBg_1.wav",
      "timestamps": [114.0, 127.0],     <- loader clips video/audio to this segment
      "text":       "..."               <- caption, kept for reference
    }
    """
    record: Dict[str, Any] = {}

    # ---- identity -----------------------------------------------------------
    record["id"] = src.get("id", "")

    # ---- type ---------------------------------------------------------------
    # "retrieval" is the default in LazySupervisedDataset; keeps InfoNCE losses active.
    record["type"] = src.get("type", "retrieval")

    # ---- conversations ------------------------------------------------------
    convs = src.get("conversations", [])
    # Ensure the human turn uses <video> token (not <image>)
    normalised_convs = []
    for turn in convs:
        t = dict(turn)
        if t.get("from") == "human":
            val = t.get("value", "")
            # Replace <image> with <video> when the record has a video field
            if "<image>" in val and "video" in src:
                val = val.replace("<image>", "<video>")
            # Ensure the <video> tag is on its own line at the start
            if "video" in src and not val.startswith("<video>"):
                val = "<video>\n" + val.lstrip()
            t["value"] = val
        normalised_convs.append(t)
    record["conversations"] = normalised_convs

    # ---- media paths --------------------------------------------------------
    # Keep as bare filenames; VIDEO_ROOT / AUDIO_ROOT are prepended by the loader.
    if "video" in src:
        record["video"] = src["video"]

    if "audio" in src:
        record["audio"] = src["audio"]

    if "image" in src:
        record["image"] = src["image"]

    if "frame_dir" in src:
        record["frame_dir"] = src["frame_dir"]

    # ---- timestamps ---------------------------------------------------------
    # Each YouCookII record is already ONE segment; timestamps tell the loader
    # which slice of the full video/audio to decode -- no further splitting needed.
    if "timestamps" in src:
        ts = src["timestamps"]
        if isinstance(ts, (list, tuple)) and len(ts) == 2:
            record["timestamps"] = [float(ts[0]), float(ts[1])]

    # ---- caption / text reference -------------------------------------------
    if "text" in src:
        record["text"] = src["text"]

    return record


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed line {lineno} in {path}: {exc}",
                      file=sys.stderr)
    return records


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Written {len(records):,} records -> {path}")


def convert(src_path: str, dst_path: str) -> None:
    print(f"\nConverting: {src_path}")
    raw = read_jsonl(src_path)
    converted = [build_omni_record(r) for r in raw]

    # Basic stats
    n_video = sum(1 for r in converted if "video" in r)
    n_audio = sum(1 for r in converted if "audio" in r)
    n_ts    = sum(1 for r in converted if "timestamps" in r)
    print(f"  Records         : {len(converted):,}")
    print(f"  With video      : {n_video:,}")
    print(f"  With audio      : {n_audio:,}")
    print(f"  With timestamps : {n_ts:,}  <- each is already one event segment")

    write_jsonl(converted, dst_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert YouCookII metadata JSONL -> train_omni.jsonl"
    )
    parser.add_argument(
        "--train",
        default="D:/Hoc/KL/Data/YouCookII/YouCookII/metadata/youcookii_train.jsonl",
        help="Path to youcookii_train.jsonl",
    )
    parser.add_argument(
        "--val",
        default="D:/Hoc/KL/Data/YouCookII/YouCookII/metadata/youcookii_val.jsonl",
        help="Path to youcookii_val.jsonl  (optional; skip if file does not exist)",
    )
    parser.add_argument(
        "--out_dir",
        default="D:/Hoc/KL/Data/YouCookII/YouCookII/metadata",
        help="Output directory for train_omni.jsonl / val_omni.jsonl",
    )
    args = parser.parse_args()

    # --- training split ---
    if os.path.isfile(args.train):
        convert(
            args.train,
            os.path.join(args.out_dir, "train_omni.jsonl"),
        )
    else:
        print(f"[ERROR] Train file not found: {args.train}", file=sys.stderr)
        sys.exit(1)

    # --- validation split (optional) ---
    if args.val and os.path.isfile(args.val):
        convert(
            args.val,
            os.path.join(args.out_dir, "val_omni.jsonl"),
        )
    else:
        print(f"[INFO] Val file not found or not provided, skipping: {args.val}")

    print("\n=== Done! ===")
    print("Next steps:")
    print("  1. Set VIDEO_ROOT and AUDIO_ROOT env vars.")
    print("  2. Pass DATA_PATH=<out_dir>/train_omni.jsonl to train.sh.")
    print("\nExample launch command:")
    print('  VIDEO_ROOT="D:/Hoc/KL/Data/YouCookII/YouCookII/videos" \\')
    print('  AUDIO_ROOT="D:/Hoc/KL/Data/YouCookII/YouCookII/audio"  \\')
    print('  WAVE_PATH="D:/Hoc/KL/Code/Omni/WAVE_HOME/WAVE-7B"      \\')
    print('  BEATS_PATH="D:/Hoc/KL/Code/Omni/WAVE_HOME/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" \\')
    print('  DATA_PATH="<out_dir>/train_omni.jsonl"                  \\')
    print('  OUTPUT_DIR="D:/Hoc/KL/Code/Omni/output/omniretriever_youcook" \\')
    print('  bash training/train.sh')


if __name__ == "__main__":
    main()
