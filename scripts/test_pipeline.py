"""
test_pipeline.py  –  Kiểm tra pipeline load dữ liệu YouCookII
Không cần GPU, không cần load model (chỉ cần Processor).

Chạy:
    $env:PYTHONIOENCODING="utf-8"
    cd D:\Học\KL\Code\Omni\Omni-fix\training
    & "D:\miniconda\conda-envs\omni\python.exe" ..\scripts\test_pipeline.py
"""

import os, sys, time, random, traceback
from pathlib import Path

# ── Đường dẫn ──────────────────────────────────────────────────────────────
# Auto-resolve từ vị trí script: Omni-fix/scripts/ → Omni-fix/ → Omni/ → Code/ → KL/
# Cấu trúc thực tế: D:\Học\KL\Code\Omni\Omni-fix\scripts
_SCRIPTS_DIR = Path(__file__).resolve().parent      # .../Omni-fix/scripts
_REPO_ROOT   = _SCRIPTS_DIR.parent                 # .../Omni-fix
_OMNI_ROOT   = _REPO_ROOT.parent                   # .../Code/Omni (cùng cấp WAVE_HOME)
_CODE_ROOT   = _OMNI_ROOT.parent                   # .../KL/Code
_KL_ROOT     = _CODE_ROOT.parent                   # .../Học/KL  (cùng cấp với Data)

REPO_ROOT  = _REPO_ROOT / "training"
WAVE_BASE  = _OMNI_ROOT / "WAVE_HOME" / "WAVE-7B"
DATA_JSONL = _KL_ROOT   / "Data" / "YouCookII" / "YouCookII" / "metadata" / "train_omni.jsonl"
VIDEO_ROOT = _KL_ROOT   / "Data" / "YouCookII" / "YouCookII" / "videos"
AUDIO_ROOT = _KL_ROOT   / "Data" / "YouCookII" / "YouCookII" / "audio"

# Resolve và set env vars
VIDEO_ROOT = str(VIDEO_ROOT.resolve())
AUDIO_ROOT = str(AUDIO_ROOT.resolve())
os.environ["VIDEO_ROOT"] = VIDEO_ROOT
os.environ["AUDIO_ROOT"] = AUDIO_ROOT
os.environ["DATA_LOADING_LOG"] = str(Path(__file__).parent.parent / "tmp" / "data_loading.log")

sys.path.insert(0, str(REPO_ROOT))

N_SAMPLES   = 5    # số mẫu kiểm tra (thay đổi nếu muốn)
SEED        = 42
OFFSET      = 0    # index bắt đầu trong dataset (0 = từ đầu)

# ── Màu sắc terminal ───────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"{GREEN}  ✓ {msg}{RESET}")
def fail(msg): print(f"{RED}  ✗ {msg}{RESET}")
def warn(msg): print(f"{YELLOW}  ! {msg}{RESET}")
def info(msg): print(f"{CYAN}  » {msg}{RESET}")

print(f"\n{BOLD}{'='*60}")
print(f"  YouCookII Data Pipeline Test")
print(f"{'='*60}{RESET}")

# ── 1. Kiểm tra thư mục / file ────────────────────────────────────────────
print(f"\n{BOLD}[1] Kiểm tra đường dẫn{RESET}")
checks = {
    "WAVE_BASE (Processor)": WAVE_BASE.resolve(),
    "DATA_JSONL":            DATA_JSONL.resolve(),
    "VIDEO_ROOT":            Path(VIDEO_ROOT),
    "AUDIO_ROOT":            Path(AUDIO_ROOT),
}
all_ok = True
for name, p in checks.items():
    if p.exists():
        ok(f"{name}: {p}")
    else:
        fail(f"{name}: {p}  ← KHÔNG TỒN TẠI")
        all_ok = False

if not all_ok:
    print(f"\n{RED}Dừng lại — một số đường dẫn không tồn tại. Hãy kiểm tra lại.{RESET}")
    sys.exit(1)

# ── 2. Load Processor (không cần model) ───────────────────────────────────
print(f"\n{BOLD}[2] Load Processor{RESET}")
t0 = time.time()
from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor
processor = Qwen2_5OmniProcessor.from_pretrained(str(WAVE_BASE.resolve()))
ok(f"Processor loaded in {time.time()-t0:.1f}s")

# ── 3. Tạo DataArgs giả (mock) ────────────────────────────────────────────
print(f"\n{BOLD}[3] Tạo DataArgs mock{RESET}")
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MockDataArgs:
    dataset_use: str = str(DATA_JSONL.resolve())
    omni_processor: object = None
    image_processor: object = None
    audio_processor: object = None
    video_max_frames: int = 8
    video_min_frames: int = 8
    base_interval: float = 0.5
    max_pixels: int = 50176
    min_pixels: int = 50176
    image_max_frame_pixels: int = 2073600
    image_min_frame_pixels: int = 784
    fixed_audio_duration: float = 8.0
    pred_embeds: bool = True
    train_classify: bool = True
    classify_type: str = "all_layer"
    use_beats: bool = True
    beats_only: bool = False
    use_tuple_infonce: bool = True
    run_test: bool = False
    do_sample: bool = False
    num_sample: int = 1
    feature_size: int = 128
    chunk_length: int = 30
    hop_length: int = 160
    sampling_rate: int = 16000
    video_max_total_pixels: int = 1664 * 28 * 28
    video_min_total_pixels: int = 256 * 28 * 28

data_args = MockDataArgs()
data_args.omni_processor = processor
ok("DataArgs mock created")

# ── 4. Khởi tạo Dataset ───────────────────────────────────────────────────
print(f"\n{BOLD}[4] Khởi tạo LazySupervisedDataset{RESET}")
t0 = time.time()
from qwenvl.data.data_qwen import LazySupervisedDataset
from transformers import AutoTokenizer

tokenizer = processor.tokenizer
dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
ok(f"Dataset initialized in {time.time()-t0:.1f}s | {len(dataset)} samples total")

# ── 5. Kiểm tra metadata một vài mẫu ─────────────────────────────────────
print(f"\n{BOLD}[5] Kiểm tra metadata{RESET}")
sample_meta = dataset.list_data_dict[0]
info(f"Sample[0] keys: {list(sample_meta.keys())}")
info(f"  type       : {sample_meta.get('type')}")
info(f"  video      : {sample_meta.get('video')}")
info(f"  audio      : {sample_meta.get('audio')}")
info(f"  timestamps : {sample_meta.get('timestamps')}")
info(f"  conv[0]    : {sample_meta['conversations'][0]['from']} → {sample_meta['conversations'][0]['value'][:80]!r}")

# ── 6. Load N_SAMPLES thực tế (video + audio decode) ──────────────────────
print(f"\n{BOLD}[6] Load {N_SAMPLES} mẫu thực tế (video decode + audio processing){RESET}")

random.seed(SEED)
indices = list(range(OFFSET, min(OFFSET + N_SAMPLES * 3, len(dataset))))[:N_SAMPLES]

results = {"ok": 0, "fail": 0, "errors": []}

for idx in indices:
    meta = dataset.list_data_dict[idx]
    vid  = meta.get("video", "?")
    ts   = meta.get("timestamps", None)
    aud  = meta.get("audio", "?")

    t0 = time.time()
    try:
        # Gọi _get_item trực tiếp để bỏ qua signal.SIGALRM (không hỗ trợ Windows)
        sample = dataset._get_item(idx)
        elapsed = time.time() - t0

        if sample is None:
            warn(f"[{idx:5d}] {vid} ts={ts} → sample=None (skipped by dataset)")
            results["fail"] += 1
            results["errors"].append(f"[{idx}] sample is None")
            continue

        # In thông tin tensor
        vid_shape  = sample.get("pixel_values_videos", [None])[0]
        vid_shape  = vid_shape.shape if vid_shape is not None else "N/A"
        aud_feat   = sample.get("input_features", None)
        aud_shape  = aud_feat.shape if aud_feat is not None else "N/A"
        tok_shape  = sample.get("input_ids", None)
        tok_shape  = tok_shape.shape if tok_shape is not None else "N/A"

        ok(f"[{idx:5d}] {vid} ts={ts} [{elapsed:.1f}s]")
        info(f"         video_tensor : {vid_shape}")
        info(f"         audio_feat   : {aud_shape}")
        info(f"         input_ids    : {tok_shape}")
        results["ok"] += 1

    except Exception as e:
        elapsed = time.time() - t0
        fail(f"[{idx:5d}] {vid} ts={ts} [{elapsed:.1f}s] ERROR: {e}")
        results["fail"] += 1
        results["errors"].append(f"[{idx}] {type(e).__name__}: {e}")
        if os.environ.get("VERBOSE_ERRORS", "0") == "1":
            traceback.print_exc()

# ── 7. Tóm tắt ─────────────────────────────────────────────────────────────
print(f"\n{BOLD}{'='*60}")
print(f"  KẾT QUẢ KIỂM TRA")
print(f"{'='*60}{RESET}")
print(f"  Tổng mẫu kiểm tra : {N_SAMPLES}")
print(f"  {GREEN}✓ Thành công       : {results['ok']}{RESET}")
print(f"  {RED}✗ Thất bại         : {results['fail']}{RESET}")

if results["errors"]:
    print(f"\n{YELLOW}  Lỗi chi tiết:{RESET}")
    for e in results["errors"]:
        print(f"    {RED}- {e}{RESET}")

if results["fail"] == 0:
    print(f"\n{GREEN}{BOLD}  ✓ Pipeline load data OK! Sẵn sàng để training.{RESET}")
else:
    print(f"\n{YELLOW}{BOLD}  ! Một số mẫu bị lỗi – xem chi tiết ở trên.{RESET}")
    print(f"  Thêm VERBOSE_ERRORS=1 để xem full traceback:")
    print(f"  $env:VERBOSE_ERRORS='1'; python scripts/test_pipeline.py")
