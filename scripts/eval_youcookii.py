#!/usr/bin/env python3
"""
Evaluation script for OmniRetriever on YouCookII Validation Set (val_omni.jsonl).
Supports Text-to-Video/Audio (t2m) and Video/Audio-to-Text (m2t) retrieval metrics:
- Recall@1, Recall@5, Recall@10
- Mean Reciprocal Rank (MRR)
- Median Rank

Usage:
    # Đánh giá zero-shot với adapter gốc:
    python scripts/eval_youcookii.py \
        --adapter "D:/Học/KL/Code/Omni/adapters/omniretriever-7b"

    # Đánh giá checkpoint sau khi fine-tune:
    python scripts/eval_youcookii.py \
        --adapter "D:/Học/KL/Code/Omni/Omni-fix/training/output/omniretriever_7b" \
        --max-samples 500   # hoặc bỏ --max-samples để chạy toàn bộ 3110 mẫu
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
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from qwenvl.data.data_qwen import LazySupervisedDataset, DataCollatorForOmniDataset
from qwenvl.model.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from qwenvl.model.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
from peft import PeftModel
from omniretriever.evaluation.metrics import all_metrics
from omniretriever.evaluation.score import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OmniRetriever on YouCookII val_omni.jsonl")
    parser.add_argument("--base-model", type=str, default="D:/Học/KL/Code/Omni/WAVE_HOME/WAVE-7B")
    parser.add_argument("--beats-path", type=str, default="D:/Học/KL/Code/Omni/WAVE_HOME/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    parser.add_argument("--adapter", type=str, default="D:/Học/KL/Code/Omni/adapters/omniretriever-7b")
    parser.add_argument("--val-manifest", type=str, default="D:/Học/KL/Data/YouCookII/YouCookII/metadata/val_omni.jsonl")
    parser.add_argument("--video-root", type=str, default="D:/Học/KL/Data/YouCookII/YouCookII/videos")
    parser.add_argument("--audio-root", type=str, default="D:/Học/KL/Data/YouCookII/YouCookII/audio")
    parser.add_argument("--output", type=str, default="./output/eval_youcookii_results.json")
    parser.add_argument("--save-embeds", type=str, default=None, help="Optional path to save embeddings .npz")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of validation samples for quick check")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_available() else "float32")
    return parser.parse_args()


class MockDataArgs:
    def __init__(self, manifest_path, processor, max_samples=None):
        self.dataset_use = manifest_path
        self.omni_processor = processor
        self.image_processor = processor.image_processor
        self.video_max_frames = 8
        self.video_min_frames = 8
        self.base_interval = 0.5
        self.max_pixels = 50176
        self.min_pixels = 50176
        self.image_max_frame_pixels = 2073600
        self.image_min_frame_pixels = 784
        self.fixed_audio_duration = 8.0
        self.pred_embeds = True
        self.train_classify = True
        self.classify_type = "all_layer"
        self.use_beats = True
        self.beats_only = False
        self.use_tuple_infonce = False
        self.run_test = False
        self.do_sample = False
        self.num_sample = 1
        self.feature_size = 128
        self.chunk_length = 30
        self.hop_length = 160
        self.sampling_rate = 16000
        self.max_samples = max_samples


def main():
    args = parse_args()
    os.environ["VIDEO_ROOT"] = args.video_root
    os.environ["AUDIO_ROOT"] = args.audio_root
    os.environ["BEATS_PATH"] = args.beats_path

    print("=" * 65)
    print("  OmniRetriever Evaluation on YouCookII Validation Set")
    print("=" * 65)
    print(f"Base Model    : {args.base_model}")
    print(f"Adapter       : {args.adapter}")
    print(f"Val Manifest  : {args.val_manifest}")
    print(f"Device / Dtype: {args.device} / {args.dtype}")
    print("=" * 65)

    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else (torch.float16 if args.dtype == "float16" else torch.float32)

    # 1. Load Processor
    print("\n[1/4] Loading Processor...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.base_model)
    tokenizer = processor.tokenizer

    # 2. Load Model
    print("\n[2/4] Loading Model & Adapter...")
    model_config = Qwen2_5OmniThinkerConfig.from_pretrained(args.base_model)
    if hasattr(model_config, "text_config"):
        if getattr(model_config.text_config, "pad_token_id", None) is None:
            model_config.text_config.pad_token_id = getattr(model_config, "pad_token_id", 151643)
        if getattr(model_config.text_config, "bos_token_id", None) is None:
            model_config.text_config.bos_token_id = getattr(model_config, "bos_token_id", 151644)
        if getattr(model_config.text_config, "eos_token_id", None) is None:
            model_config.text_config.eos_token_id = getattr(model_config, "eos_token_id", 151645)
    model_config.train_classify = True
    model_config.classify_type = "all_layer"
    model_config.audio_config.beats_path = args.beats_path
    model_config.audio_config.beats_only = False

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.base_model,
        config=model_config,
        torch_dtype=torch_dtype,
    )
    if os.path.exists(args.beats_path):
        beats_ckpt = torch.load(args.beats_path, map_location="cpu", weights_only=False)
        model.beats.load_state_dict(beats_ckpt["model"])

    if args.adapter and args.adapter != "No":
        print(f"Applying LoRA adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)

    model = model.to(args.device).eval()

    # 3. Load Validation Dataset
    print("\n[3/4] Preparing Dataset...")
    data_args = MockDataArgs(args.val_manifest, processor, max_samples=args.max_samples)
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)

    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset.list_data_dict = dataset.list_data_dict[:args.max_samples]
        print(f"Subsampled to {len(dataset)} items.")
    else:
        print(f"Total validation samples: {len(dataset)}")

    collator = DataCollatorForOmniDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # 4. Extract Embeddings
    print("\n[4/4] Extracting paired text & multimodal embeddings...")
    text_embeds_list = []
    mllm_embeds_list = []
    ids_list = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Extracting"):
            if batch is None:
                continue

            batch_ids = batch.get("id", [])
            ids_list.extend(batch_ids)

            # Move inputs to device
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)

            pixel_values_videos = [v.to(args.device, dtype=torch_dtype) for v in batch["pixel_values_videos"]] if batch.get("pixel_values_videos") else None
            video_grid_thw = [t.to(args.device) for t in batch["video_grid_thw"]] if batch.get("video_grid_thw") else None
            video_second_per_grid = batch.get("video_second_per_grid")

            input_features = [a.to(args.device, dtype=torch_dtype) for a in batch["input_features"]] if batch.get("input_features") else None
            feature_attention_mask = [m.to(args.device) for m in batch["feature_attention_mask"]] if batch.get("feature_attention_mask") else None
            input_raw_wav = batch.get("input_raw_wav")

            # Forward pass to get embeddings
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                video_second_per_grid=video_second_per_grid,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                input_raw_wav=input_raw_wav,
                pred_embeds=True,
                return_dict=True,
            )

            # Extract normalized embeddings
            if hasattr(outputs, "mllm_embeds") and outputs.mllm_embeds is not None:
                m_emb = outputs.mllm_embeds.cpu().float()
                m_emb = m_emb / torch.norm(m_emb, dim=-1, keepdim=True).clamp(min=1e-12)
                mllm_embeds_list.append(m_emb)

            if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                t_emb = outputs.text_embeds.cpu().float()
                t_emb = t_emb / torch.norm(t_emb, dim=-1, keepdim=True).clamp(min=1e-12)
                text_embeds_list.append(t_emb)

    text_all = torch.cat(text_embeds_list, dim=0).numpy()
    mllm_all = torch.cat(mllm_embeds_list, dim=0).numpy()

    print(f"\nExtracted shapes: Text={text_all.shape}, Multimodal={mllm_all.shape}")

    if args.save_embeds:
        save_p = Path(args.save_embeds)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_p, text_embeds=text_all, mllm_embeds=mllm_all, ids=ids_list)
        print(f"Saved embeddings to {save_p}")

    # Compute similarity matrix (N x N)
    sim = cosine_similarity(text_all, mllm_all)

    metrics_t2m = all_metrics(sim)       # Text -> Video/Audio
    metrics_m2t = all_metrics(sim.T)     # Video/Audio -> Text

    results = {
        "text_to_multimodal (t2m)": {k: round(v * 100 if k.startswith("R@") or k == "MRR" else v, 2) for k, v in metrics_t2m.items()},
        "multimodal_to_text (m2t)": {k: round(v * 100 if k.startswith("R@") or k == "MRR" else v, 2) for k, v in metrics_m2t.items()},
        "num_samples": len(text_all),
    }

    print("\n" + "=" * 65)
    print("               KẾT QUẢ ĐÁNH GIÁ YOUCOOKII (VAL)")
    print("=" * 65)
    print(f"{'Chỉ số':<15} | {'Text -> Video (t2m)':<22} | {'Video -> Text (m2t)':<22}")
    print("-" * 65)
    print(f"{'Recall@1':<15} | {results['text_to_multimodal (t2m)']['R@1']:>18.2f}% | {results['multimodal_to_text (m2t)']['R@1']:>18.2f}%")
    print(f"{'Recall@5':<15} | {results['text_to_multimodal (t2m)']['R@5']:>18.2f}% | {results['multimodal_to_text (m2t)']['R@5']:>18.2f}%")
    print(f"{'Recall@10':<15} | {results['text_to_multimodal (t2m)']['R@10']:>18.2f}% | {results['multimodal_to_text (m2t)']['R@10']:>18.2f}%")
    print(f"{'MRR':<15} | {results['text_to_multimodal (t2m)']['MRR']:>18.2f}% | {results['multimodal_to_text (m2t)']['MRR']:>18.2f}%")
    print(f"{'Median Rank':<15} | {results['text_to_multimodal (t2m)']['median_rank']:>20} | {results['multimodal_to_text (m2t)']['median_rank']:>20}")
    print("=" * 65)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nChi tiết kết quả đã được lưu tại: {out_p}")


if __name__ == "__main__":
    main()
