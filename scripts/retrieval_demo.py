#!/usr/bin/env python3
"""
Interactive / CLI Multimodal Retrieval Demo with OmniRetriever.

Cho phép bạn:
1. Nhập câu text truy vấn -> Tìm Top-K video / đoạn clip phù hợp nhất trong YouCookII.
2. Đưa vào file video -> Tìm Top-K câu mô tả (caption) phù hợp nhất.

Cách dùng:
    # Bước 1: Trích xuất và lưu gallery embeddings của tập val (chỉ cần chạy 1 lần):
    python scripts/eval_youcookii.py --save-embeds output/val_embeds.npz --max-samples 500

    # Bước 2: Truy vấn Text -> Top-K Video:
    python scripts/retrieval_demo.py --query "place cheese on the bread" --top-k 5

    # Hoặc chạy chế độ tương tác (nhập câu hỏi liên tục):
    python scripts/retrieval_demo.py --interactive
"""

# Tránh lỗi circular import giữa deepspeed và transformers.modeling_utils
try:
    import deepspeed
except Exception:
    pass
import transformers

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from omniretriever import OmniRetriever


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Retrieval Demo")
    parser.add_argument("--base-model", type=str, default="D:/Học/KL/Code/Omni/WAVE_HOME/WAVE-7B")
    parser.add_argument("--adapter", type=str, default="D:/Học/KL/Code/Omni/adapters/omniretriever-7b")
    parser.add_argument("--val-manifest", type=str, default="D:/Học/KL/Data/YouCookII/YouCookII/metadata/val_omni.jsonl")
    parser.add_argument("--gallery-embeds", type=str, default="output/val_embeds.npz", help="Precomputed gallery embeddings from eval_youcookii.py")
    parser.add_argument("--query", type=str, default=None, help="Text query to search for videos")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive prompt mode")
    return parser.parse_args()


def load_gallery(embeds_path, manifest_path):
    if not os.path.exists(embeds_path):
        print(f"[-] Chưa tìm thấy file embeddings: {embeds_path}")
        print(f"    Hãy chạy lệnh sau để trích xuất trước:")
        print(f"    python scripts/eval_youcookii.py --save-embeds {embeds_path}")
        sys.exit(1)

    print(f"[+] Loading gallery embeddings from {embeds_path}...")
    blob = np.load(embeds_path, allow_pickle=True)
    mllm_embeds = blob["mllm_embeds"]  # shape (N, 3584)
    ids = list(blob["ids"]) if "ids" in blob else []

    print(f"[+] Loading metadata from {manifest_path}...")
    metadata_map = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                metadata_map[item["id"]] = item

    records = [metadata_map.get(rec_id, {}) for rec_id in ids]
    return mllm_embeds, records


def search_text_to_video(query_text, model, gallery_embeds, records, top_k=5):
    print(f"\n[Query] \"{query_text}\"")
    # 1. Trích xuất embedding câu truy vấn
    z_query = model.encode_text(query_text).cpu().float().numpy()
    z_query = z_query / np.linalg.norm(z_query, axis=-1, keepdims=True)

    # 2. Tính Cosine Similarity với toàn bộ kho video
    scores = (z_query @ gallery_embeds.T).flatten()

    # 3. Lấy Top-K index có điểm cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]

    print(f"\n{'Hạng':<5} | {'Score':<8} | {'Video File':<25} | {'Timestamps':<15} | {'Ground Truth Caption'}")
    print("-" * 90)
    for rank, idx in enumerate(top_indices, 1):
        rec = records[idx]
        score = scores[idx]
        vid = rec.get("video", "N/A")
        ts = str(rec.get("timestamps", "N/A"))
        caption = rec.get("text", "")
        print(f"{rank:<5} | {score:>7.4f} | {vid:<25} | {ts:<15} | {caption}")
    print("-" * 90)


def main():
    args = parse_args()
    gallery_embeds, records = load_gallery(args.gallery_embeds, args.val_manifest)

    print(f"[+] Loading OmniRetriever model...")
    model = OmniRetriever.from_pretrained(
        base_model=args.base_model,
        adapter=args.adapter,
        device=args.device,
        dtype=args.dtype,
    )
    print(f"[+] Model loaded successfully on {args.device} ({args.dtype}).")

    if args.query:
        search_text_to_video(args.query, model, gallery_embeds, records, top_k=args.top_k)

    if args.interactive or not args.query:
        print("\n=== CHẾ ĐỘ TRUY VẤN TƯƠNG TÁC (Gõ 'exit' hoặc 'quit' để thoát) ===")
        while True:
            try:
                q = input("\nNhập câu mô tả món ăn / hành động cần tìm video: ").strip()
                if not q:
                    continue
                if q.lower() in ("exit", "quit", "q"):
                    break
                search_text_to_video(q, model, gallery_embeds, records, top_k=args.top_k)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
