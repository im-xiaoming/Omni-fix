#!/usr/bin/env python3
"""
convert_youcookii.py
====================
Build the video-only YouCookII manifests that training and inference both read.

The only media input is the full ``<video_id>.mp4``. Each record is one event
(one recipe step) from the metadata: its ``timestamps`` are the ``[start, end]``
segment, and at load time the event is cut out of the video on the fly --

* the frames by ``training/qwenvl/data/data_qwen.py::video_decord`` (training) /
  ``omniretriever.data.media.load_video_frames`` (inference), and
* the audio by ``omniretriever.data.media.load_audio_segment``, from the same
  mp4 over the same window, in both training and inference.

No clip or wav is written to disk, and records carry no ``audio`` field.

Input is ``youcookii_{train,val}_preprocess.json`` (``{"database": {event_id:
{video_id, segment, sentence, duration, ...}}}``). Events whose video is not
under ``--video-root`` are dropped and counted.

Segments are checked against the length of the video actually on disk, read
from its header, not against the metadata's ``duration``: some downloads are
shorter than the original upload (``hs2h7nb5PHQ`` is 215.8 s on disk, 316.8 s
in the metadata). An event that starts less than ``MIN_EVENT_SEC`` before the
file ends is dropped -- decord turns such a window into out-of-bound frame
indices and the audio cut comes back empty -- and an event that runs past the
end is clamped to it.

Usage
-----
    python scripts/convert_youcookii.py \\
        --metadata-dir "D:/Học/KL/Data/YouCookII/metadata" \\
        --video-root   "D:/Học/KL/Data/YouCookII/videos"

writes ``train_omni_video.jsonl`` and ``val_omni_video.jsonl`` into
``--metadata-dir``. Train with ``DATA_PATH=.../train_omni_video.jsonl`` and
``VIDEO_ROOT=.../videos`` (``AUDIO_ROOT`` is no longer used).

Output record
-------------
    {
      "id": "GLd3aX16zBg_1",
      "type": "retrieval",
      "conversations": [
        {"from": "human", "value": "<video>\\nPlease describe the video."},
        {"from": "gpt",   "value": "place a slice of cheese on the bread"}
      ],
      "video": "GLd3aX16zBg.mp4",        <- bare filename; $VIDEO_ROOT prepended at load time
      "timestamps": [114.0, 127.0],      <- the event; frames AND audio are cut to it
      "text": "place a slice of cheese on the bread"
    }
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROMPT = "<video>\nPlease describe the video."
SPLITS = ("train", "val")
# Shortest event kept after clamping to the file: one second, the floor the
# training loader pads audio to anyway.
MIN_EVENT_SEC = 1.0


def probe_duration(path: str) -> Optional[float]:
    """Seconds of the file that both streams cover, from the header only.

    The shorter of the video and audio stream: a frame window past the audio's
    end would pair with a silent tail. ``None`` when the file cannot be opened
    or has no audio stream, which the loader cannot cut an event's audio from.
    """
    import av

    try:
        with av.open(path) as container:
            if not container.streams.video or not container.streams.audio:
                return None
            spans = [
                float(s.duration * s.time_base)
                for s in (container.streams.video[0], container.streams.audio[0])
                if s.duration and s.time_base
            ]
            if not spans and container.duration:
                spans = [container.duration / 1e6]
            return min(spans) if spans else None
    except Exception:  # noqa: BLE001 - PyAV raises several error types
        return None


def probe_durations(video_root: str, names: Iterable[str], workers: int) -> Dict[str, Optional[float]]:
    names = sorted(set(names))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        spans = pool.map(lambda n: probe_duration(os.path.join(video_root, n)), names)
        return dict(zip(names, spans))


def build_record(event_id: str, event: Dict[str, Any], start: float, end: float) -> Dict[str, Any]:
    caption = event["sentence"].strip()
    return {
        "id": event_id,
        "type": "retrieval",
        "conversations": [
            {"from": "human", "value": PROMPT},
            {"from": "gpt", "value": caption},
        ],
        "video": f"{event['video_id']}.mp4",
        "timestamps": [float(start), float(end)],
        "text": caption,
    }


def convert(src_path: str, video_root: str, workers: int = 8) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    with open(src_path, "r", encoding="utf-8") as f:
        database = json.load(f)["database"]

    available = {name for name in os.listdir(video_root) if name.endswith(".mp4")}
    wanted = {f"{e['video_id']}.mp4" for e in database.values()} & available
    durations = probe_durations(video_root, wanted, workers)

    records = []
    stats = {"events": len(database), "missing_video": 0, "unreadable_video": 0,
             "bad_segment": 0, "past_video_end": 0, "clamped_end": 0}
    for event_id, event in database.items():
        name = f"{event['video_id']}.mp4"
        start, end = float(event["segment"][0]), float(event["segment"][1])
        if name not in available:
            stats["missing_video"] += 1
            continue
        if durations[name] is None:
            stats["unreadable_video"] += 1
            continue
        if end <= start:
            stats["bad_segment"] += 1
            continue
        if start > durations[name] - MIN_EVENT_SEC:
            stats["past_video_end"] += 1
            continue
        if end > durations[name]:
            stats["clamped_end"] += 1
            end = round(durations[name], 3)
        records.append(build_record(event_id, event, start, end))
    stats["written"] = len(records)
    stats["videos"] = len({r["video"] for r in records})
    return records, stats


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="YouCookII metadata -> video-only event manifests")
    parser.add_argument("--metadata-dir", default="D:/Học/KL/Data/YouCookII/metadata",
                        help="Folder holding youcookii_{train,val}_preprocess.json")
    parser.add_argument("--video-root", default="D:/Học/KL/Data/YouCookII/videos",
                        help="Folder holding the full <video_id>.mp4 files")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write {train,val}_omni_video.jsonl (default: --metadata-dir)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Threads for reading video headers (default 8; more helps on a Drive mount)")
    args = parser.parse_args()
    out_dir = args.out_dir or args.metadata_dir

    if not os.path.isdir(args.video_root):
        sys.exit(f"[ERROR] video root not found: {args.video_root}")

    for split in SPLITS:
        src = os.path.join(args.metadata_dir, f"youcookii_{split}_preprocess.json")
        if not os.path.isfile(src):
            print(f"[WARN] {src} not found, skipping {split}")
            continue
        records, stats = convert(src, args.video_root, args.workers)
        dst = os.path.join(out_dir, f"{split}_omni_video.jsonl")
        write_jsonl(records, dst)
        print(f"{split:5s}: {stats['written']:,} events from {stats['videos']:,} videos -> {dst}"
              f"  (of {stats['events']:,}; missing video {stats['missing_video']},"
              f" unreadable/no audio {stats['unreadable_video']},"
              f" bad segment {stats['bad_segment']},"
              f" starts past video end {stats['past_video_end']},"
              f" end clamped {stats['clamped_end']})")


if __name__ == "__main__":
    main()
