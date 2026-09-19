# HƯỚNG DẪN CHI TIẾT: FINE-TUNE, EVAL VÀ RETRIEVAL (OMNIRETRIEVER-7B)

Tài liệu này hướng dẫn toàn bộ quy trình thực hiện với mô hình **OmniRetriever-7B** trên bộ dữ liệu **YouCookII**:
1. **Fine-tuning**: Huấn luyện thích nghi miền từ adapter có sẵn.
2. **Evaluation**: Đánh giá các chỉ số truy vấn (Recall@1/5/10, MRR, Median Rank) trên tập Validation (`val_omni_video.jsonl`).
3. **Retrieval**: Thực hiện tìm kiếm video từ văn bản (Text $\rightarrow$ Video) hoặc ngược lại.

---

## 0. Bảng cấu hình đường dẫn chuẩn

Toàn bộ script trong repository đã được cấu hình sẵn các đường dẫn mặc định sau:

| Tên biến | Đường dẫn trên máy | Mô tả |
| :--- | :--- | :--- |
| **`WAVE_PATH`** | `D:\Học\KL\Code\Omni\WAVE_HOME\WAVE-7B` | Trọng số mô hình nền WAVE-7B |
| **`BEATS_PATH`** | `D:\Học\KL\Code\Omni\WAVE_HOME\BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` | Checkpoint Audio Encoder BEATs |
| **`DATA_PATH`** | `D:\Học\KL\Data\YouCookII\metadata\train_omni_video.jsonl` | Manifest huấn luyện, chỉ có video (8,560 event) |
| **`VAL_MANIFEST`** | `D:\Học\KL\Data\YouCookII\metadata\val_omni_video.jsonl` | Manifest validation, chỉ có video (3,030 event) |
| **`VIDEO_ROOT`** | `D:\Học\KL\Data\YouCookII\videos` | Thư mục chứa các video gốc `<video_id>.mp4` |
| ~~`AUDIO_ROOT`~~ | không còn dùng | Audio của từng event được cắt từ chính video (xem mục 0.1) |
| **`LORA_CKPT`** | `D:\Học\KL\Code\Omni\adapters\omniretriever-7b` | Adapter LoRA gốc của OmniRetriever |
| **`OUTPUT_DIR`** | `D:\Học\KL\Code\Omni\Omni-fix\training\output\omniretriever_7b` | Thư mục lưu checkpoint sau khi train |

### 0.1 Đầu vào chỉ là video: cắt event theo metadata

Đầu vào duy nhất là video gốc `<video_id>.mp4`. Mỗi **event** (một bước nấu ăn trong metadata) là một record
có `timestamps = [start, end]`. Lúc nạp dữ liệu, event được cắt ra **ngay trong bộ nhớ**, không ghi clip hay
file `.wav` nào ra đĩa:

* **Frames**: lấy mẫu 8 frame trong cửa sổ `[start, end]` của video.
* **Audio**: cắt track audio của **chính video đó** trên **cùng cửa sổ** `[start, end]`
  (`omniretriever.data.media.load_audio_segment`: seek tới `start`, decode tới `end`, trộn mono, resample 16 kHz),
  rồi center-crop/pad về 8 s (`fit_waveform`).

Training (`training/qwenvl/data/data_qwen.py`) và inference (`src/omniretriever/...`, `scripts/eval_youcookii.py`)
dùng **chung** hai hàm cắt audio trên, nên event được cắt giống hệt nhau ở cả hai phía. Đã kiểm chứng: audio cắt
theo cách này trùng với các file `.wav` cắt sẵn trong `Data/YouCookII/audio` (tương quan ≥ 0.99, lệch 0 ms, đúng độ dài).

Sinh manifest (chạy một lần, chỉ đọc header video, không cần GPU):

```powershell
python scripts/convert_youcookii.py `
    --metadata-dir "D:\Học\KL\Data\YouCookII\metadata" `
    --video-root   "D:\Học\KL\Data\YouCookII\videos"
```

Script đối chiếu `segment` với **độ dài thật** của file video trên đĩa (một số video tải về ngắn hơn metadata):
event bắt đầu sau khi video đã hết bị loại, event chạy quá cuối video được cắt về cuối video. Kết quả:

| Split | Event trong metadata | Thiếu video | Bắt đầu sau cuối video | Clamp `end` | Ghi ra |
| :-- | --: | --: | --: | --: | --: |
| train | 8,815 | 252 | 3 | 1 | **8,560** |
| val | 3,109 | 78 | 1 | 0 | **3,030** |

Record mẫu (không có trường `audio`):

```json
{"id": "GLd3aX16zBg_1", "type": "retrieval",
 "conversations": [{"from": "human", "value": "<video>\nPlease describe the video."},
                   {"from": "gpt", "value": "place a slice of cheese on the bread"}],
 "video": "GLd3aX16zBg.mp4", "timestamps": [114.0, 127.0],
 "text": "place a slice of cheese on the bread"}
```

Kiểm tra trước khi train (đọc header, không cần GPU) — cả hai split phải báo coverage 100%:

```bash
VIDEO_ROOT=/path/to/YouCookII/videos python scripts/check_tuple_data.py /path/to/metadata/train_omni_video.jsonl
```

Trên Colab, chạy thêm `scripts/test_pipeline.py` (chỉ cần processor, không nạp trọng số) để xem tensor video/audio
thật của vài event: `WAVE_PATH=... DATA_PATH=... VIDEO_ROOT=... python scripts/test_pipeline.py`.

> Manifest có trường `audio` (layout cũ `train_omni.jsonl` + thư mục `audio/`) vẫn chạy được: record có `audio`
> đọc file đó, record không có `audio` thì cắt từ video.

---

## 1. Giai đoạn 1: Huấn luyện (Fine-tuning)

### Cơ chế đóng băng mô hình (Freezing & Unfreezing)
* **Luôn đóng băng**: LLM Qwen2.5 (kể cả `lm_head`, embedding), Vision Tower (ViT) + merger, Audio Tower (Whisper), BEATs encoder.
* **Phần được train** do biến `LORA_ONLY` quyết định (fine-tune tiếp từ adapter `omniretriever-7b`):

| Module | `LORA_ONLY=False` (mặc định) | `LORA_ONLY=True` | Số tham số |
| :-- | :-- | :-- | --: |
| LoRA `q/k/v_proj`, 28 layer LLM (r=16, α=32 lấy từ `adapter_config.json` của adapter) | train | train | 6,881,280 |
| `classify_linear` — fusion head all-layer: Linear(28×3584→3584) → GELU → Linear(3584→3584) | train | giữ trọng số adapter | 372,513,792 |
| `beats_proj` — Linear(768→3584) → GELU → Linear(3584→3584) | train | giữ trọng số adapter | 15,604,736 |
| `beats_ln` — LayerNorm(768) | train | giữ trọng số adapter | 1,536 |
| **Tổng trainable** | **395,001,344 (4.03%)** | **6,881,280 (0.07%)** | |

* Ba head được train qua bản sao `modules_to_save` của peft: đó là bản mà forward dùng và `save_pretrained` ghi vào
  `adapter_model.safetensors`, nên checkpoint tự mang theo head đã fine-tune.
* `LORA_ONLY=False` tốn thêm khoảng **5 GiB VRAM** (trạng thái AdamW fp32 + gradient của ~388M tham số head). Nếu Colab
  báo OOM, giảm `BATCH_SIZE` và tăng `GRAD_ACCUM` để giữ nguyên batch hiệu dụng.
* Khi bắt đầu train, log in bảng `TRAINABLE PARAMETER REPORT`, cuối bảng có mục **Trainable parameters by group** —
  kiểm tra ở đây rằng các nhóm `LoRA`, `classify_linear`, `beats_proj`, `beats_ln` khớp với bảng trên.

> Trước bản sửa này, `LORA_ONLY=False` vẫn chỉ train LoRA: trên `PeftModel`, `model.model` trỏ tới cả Thinker nên
> `set_model` đóng băng luôn ba head. Checkpoint train trước bản sửa (ví dụ `checkpoint-1102`) có head y hệt adapter gốc.

---

### A. Chạy trên Windows (PowerShell)

Mở PowerShell tại thư mục `D:\Học\KL\Code\Omni\Omni-fix`:

```powershell
# 1. Kích hoạt môi trường Conda
conda activate omni

# 2. Thiết lập mã hóa UTF-8 (bắt buộc cho đường dẫn tiếng Việt "D:\Học\KL")
$env:PYTHONIOENCODING = "utf-8"

# 3. Bước kiểm tra trước (Dry-run: in câu lệnh và kiểm tra path, không load model)
.\training\train.ps1 -DryRun

# 4. Bắt đầu Fine-tuning thực tế
.\training\train.ps1 `
    -BatchSize 1 `
    -GradAccum 8 `
    -Epochs 3 `
    -Lr 1e-5 `
    -GradientCheckpointing `
    -OutputDir "D:\Học\KL\Code\Omni\Omni-fix\training\output\youcookii_ft"
```

* **Nếu muốn đổi path bất kỳ khi chạy lệnh**:
  ```powershell
  .\training\train.ps1 -DataPath "D:\path_khác\train.jsonl" -OutputDir "D:\path_khác\out"
  ```
* **Nếu muốn train LoRA mới từ đầu (không kế thừa adapter cũ)**:
  ```powershell
  .\training\train.ps1 -LoraCkpt "No"
  ```

---

### B. Chạy trên Linux / Server / GPU Cloud / WSL2 (Bash)

```bash
cd /path/to/Omni-fix

export VIDEO_ROOT="D:/Học/KL/Data/YouCookII/videos"
export WAVE_PATH="D:/Học/KL/Code/Omni/WAVE_HOME/WAVE-7B"
export BEATS_PATH="D:/Học/KL/Code/Omni/WAVE_HOME/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
export DATA_PATH="D:/Học/KL/Data/YouCookII/metadata/train_omni_video.jsonl"
export LORA_CKPT="D:/Học/KL/Code/Omni/adapters/omniretriever-7b"
export OUTPUT_DIR="./output/youcookii_ft"

export BATCH_SIZE=1
export GRAD_ACCUM=8
export EPOCHS=3
export LR=1e-5
export GRADIENT_CHECKPOINTING="True"

bash training/train.sh
```

---

### C. Chạy trên Google Colab

Máy local không nạp được model, nên fine-tune chạy trên Colab (GPU A100). Đường dẫn dưới đây khớp với lần chạy trước
(`log.md`); sửa `DRIVE_DATA` nếu dữ liệu của bạn nằm chỗ khác.

**Chuẩn bị ở máy local (một lần):**
1. Sinh manifest chỉ-video (mục 0.1) rồi upload `train_omni_video.jsonl`, `val_omni_video.jsonl` lên
   `.../YouCookII/metadata/` trên Drive. Thư mục `videos/` trên Drive giữ nguyên; **không cần** thư mục `audio/`.
2. Push code đã sửa lên GitHub để Colab clone về.

**Cell 1 — Drive, code, thư viện**

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone -q https://github.com/im-xiaoming/Omni-fix.git /content/code/Omni-fix
# av là BẮT BUỘC: audio của mỗi event được cắt từ video bằng PyAV (load_audio_segment).
!pip install -q deepspeed peft accelerate librosa soundfile av einops
```

**Cell 2 — Trọng số WAVE-7B + BEATs và adapter gốc**

```python
!hf download nguyenminh04/omni-model --local-dir /content/WAVE_HOME --quiet
!hf download YunzeLiu/OmniRetriever-7B --local-dir /content/adapters/omniretriever-7b --quiet
```

Cấu trúc sau khi tải phải giống `architecture/WAVE_HOME.txt` và `architecture/adapters.txt`:
`/content/WAVE_HOME/WAVE-7B/`, `/content/WAVE_HOME/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`,
`/content/adapters/omniretriever-7b/adapter_model.safetensors`.

**Cell 3 — Biến môi trường**

```python
import os
DRIVE_DATA = '/content/drive/MyDrive/Colab Notebooks/code KL/Omni/data/YouCookII'
os.environ.update({
    'WAVE_PATH':  '/content/WAVE_HOME/WAVE-7B/',
    'BEATS_PATH': '/content/WAVE_HOME/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
    'DATA_PATH':  f'{DRIVE_DATA}/metadata/train_omni_video.jsonl',
    'VIDEO_ROOT': f'{DRIVE_DATA}/videos',
    'LORA_CKPT':  '/content/adapters/omniretriever-7b',
    # Ghi checkpoint thẳng lên Drive để không mất khi Colab ngắt phiên.
    'OUTPUT_DIR': '/content/drive/MyDrive/omniretriever/youcookii_ft',
    'LORA_ONLY':  'True',    # True: chỉ LoRA | False: LoRA + classify_linear + beats_ln + beats_proj
    'BATCH_SIZE': '8', 'GRAD_ACCUM': '1', 'EPOCHS': '3', 'LR': '1e-4',
    'LORA_INIT_ONLY': 'True',  # lần chạy đầu: khởi tạo từ adapter gốc; resume sau khi ngắt phiên: 'False'
})
VAL = f'{DRIVE_DATA}/metadata/val_omni_video.jsonl'
RES = '/content/drive/MyDrive/omniretriever/eval'   # kết quả eval (JSON) của zero-shot và từng epoch
os.makedirs(RES, exist_ok=True)
```

Cấu hình trên là fine-tune **LoRA-only**: LR `1e-4` hợp với việc chỉ train 6.9M tham số LoRA trên khoảng 1–3k bước
(LR 1e-5 của bài báo dành cho 395M tham số và 11k bước), 3 epoch, lưu mỗi epoch để chọn epoch tốt nhất.

**Cell 4 — Kiểm tra dữ liệu trước khi train (không nạp model)**

```python
%cd /content/code/Omni-fix
# Coverage phải là 100%: mọi event cắt được audio từ video của nó.
!python scripts/check_tuple_data.py "$DATA_PATH" --batch-size 8 --grad-accum 1
# Nạp thử vài event qua đúng dataset của training (chỉ cần processor).
!python scripts/test_pipeline.py
```

**Cell 5 — Chấm zero-shot (mốc so sánh, chạy TRƯỚC khi train)**

Adapter gốc, chưa fine-tune, trên toàn bộ 3,030 event val. Bài báo không có số YouCook, và mô hình của họ được đánh
giá zero-shot; đây là mốc mà mọi checkpoint fine-tune phải vượt. Chạy thử `--max-samples 50` trước để bắt lỗi đường
dẫn, rồi chạy đủ (batch 1, không tăng: fusion head lấy vector ở token cuối cố định nên padding làm đổi embedding).

```python
%cd /content/code/Omni-fix
# --base-model / --beats-path / --video-root lấy từ biến môi trường ở cell 3.
!python scripts/eval_youcookii.py --adapter /content/adapters/omniretriever-7b \
    --val-manifest "{VAL}" --output "{RES}/zeroshot.json" --save-embeds "{RES}/zeroshot.npz"
```

**Cell 6 — Train**

```python
!bash training/train.sh --adam_beta2 0.95 --save_strategy epoch --save_total_limit 3
```

Không truyền `--temperature`: mặc định 0.01, đúng giá trị adapter gốc đã được train. Mỗi epoch là 1,070 bước
(8,560 event / batch 8), nên checkpoint có tên `checkpoint-1070`, `checkpoint-2140`, `checkpoint-3210`.

**Cell 7 — Chấm từng epoch và so với zero-shot**

Train và eval không chạy song song được trên cùng một GPU (mỗi bên nạp một model 7B). Chạy cell này sau khi train
xong, hoặc ở phiên Colab sau: checkpoint nằm trên Drive, cell bỏ qua các checkpoint đã chấm.

```python
import glob, json
ckpts = sorted(glob.glob(f"{os.environ['OUTPUT_DIR']}/checkpoint-*"), key=lambda p: int(p.rsplit('-', 1)[1]))
for ck in ckpts:
    out = f"{RES}/{os.path.basename(ck)}.json"
    if not os.path.exists(out):
        !python scripts/eval_youcookii.py --adapter "{ck}" --val-manifest "{VAL}" --output "{out}"

def scores(path):
    r = json.load(open(path))
    return r['text_to_multimodal (t2m)'], r['multimodal_to_text (m2t)']

zs_t2m, zs_m2t = scores(f'{RES}/zeroshot.json')
M = ('R@1', 'R@5', 'R@10', 'MRR')
print(f"{'':18}" + ''.join(f'{d + " " + m:>16}' for d in ('t2m', 'm2t') for m in M))
print(f"{'zero-shot':18}" + ''.join(f'{s[m]:>16.2f}' for s in (zs_t2m, zs_m2t) for m in M))
for ck in ckpts:
    out = f"{RES}/{os.path.basename(ck)}.json"
    if os.path.exists(out):
        t2m, m2t = scores(out)
        print(f'{os.path.basename(ck):18}' + ''.join(
            f'{s[m]:.2f} ({s[m] - z[m]:+.2f})'.rjust(16)
            for s, z in ((t2m, zs_t2m), (m2t, zs_m2t)) for m in M))
```

Chọn checkpoint theo R@1 (hai chiều t2m và m2t), số trong ngoặc là chênh lệch so với zero-shot:
* Epoch 1 tốt nhất rồi giảm → bắt đầu overfit: lấy epoch 1 (muốn thử thêm thì chạy lại với `LR=5e-5`).
* Vẫn tăng tới epoch cuối → train thêm: `LORA_INIT_ONLY=False`, tăng `EPOCHS` rồi chạy lại cell 6.
* Ngay epoch 1 đã thấp hơn zero-shot → LR quá cao: chạy lại với `LR=5e-5` hoặc `2e-5` (xóa `OUTPUT_DIR` cũ trước).

> Chọn epoch trên val rồi báo cáo luôn trên val cho số hơi lạc quan. Để chặt chẽ, tách khoảng 10% video của tập train
> làm dev để chọn epoch/LR, và chỉ chấm val một lần cho checkpoint cuối.

Ngay sau bước `[5/5] Configuring model adapters / LoRA...`, xem mục **Trainable parameters by group** trong
`TRAINABLE PARAMETER REPORT`: với `LORA_ONLY=False` phải có `LoRA 6,881,280`, `classify_linear 372,513,792`,
`beats_proj 15,604,736`, `beats_ln 1,536`; với `LORA_ONLY=True` chỉ có dòng `LoRA`. Nếu OOM, đặt
`BATCH_SIZE=4`, `GRAD_ACCUM=2` rồi chạy lại.

Phiên Colab bị ngắt thì chạy lại cell 3 và 6: có `checkpoint-*` trong `OUTPUT_DIR` là script tự resume, trừ khi
`LORA_INIT_ONLY=True` (mặc định của `train.sh`) — khi đó đặt `LORA_INIT_ONLY=False` để resume từ checkpoint cục bộ.

**Sau khi train:** chấm checkpoint bằng notebook `Omni_inference.ipynb`. Cell 6 của notebook kiểm tra checkpoint có đủ
`classify_linear`, `beats_ln`, `beats_proj`, và nếu có `/content/adapters/omniretriever-7b` thì in độ lệch của từng head
so với adapter gốc (khác 0 nghĩa là head đã được fine-tune).

---

## 2. Giai đoạn 2: Đánh giá mô hình (Evaluation)

Trong bài toán Video/Audio-Text Retrieval, việc đánh giá không dùng cross-entropy loss thông thường mà đo trực tiếp bằng các chỉ số xếp hạng tìm kiếm trên tập Validation (`val_omni_video.jsonl`):
* **Recall@1 (R@1)**: Tỷ lệ tìm đúng video clip ở vị trí đầu tiên.
* **Recall@5 (R@5)**: Tỷ lệ đúng nằm trong top 5 kết quả.
* **Recall@10 (R@10)**: Tỷ lệ đúng nằm trong top 10 kết quả.
* **Mean Reciprocal Rank (MRR)** và **Median Rank**.
* Đo độc lập 2 chiều: **Text $\rightarrow$ Video ($t2m$)** và **Video $\rightarrow$ Text ($m2t$)**.

Script đánh giá: [scripts/eval_youcookii.py](file:///d:/H%E1%BB%8Dc/KL/Code/Omni/Omni-fix/scripts/eval_youcookii.py).

### Bước 2.1: Đánh giá Zero-shot Baseline (Adapter gốc)
Đo điểm của adapter phát hành sẵn trước khi train để có mốc so sánh:

```powershell
# Chạy kiểm tra nhanh trên 100 mẫu đầu tiên:
python scripts/eval_youcookii.py `
    --adapter "D:\Học\KL\Code\Omni\adapters\omniretriever-7b" `
    --max-samples 100

# Hoặc chạy toàn bộ 3,030 event validation và lưu embedding:
python scripts/eval_youcookii.py `
    --adapter "D:\Học\KL\Code\Omni\adapters\omniretriever-7b" `
    --save-embeds "output/baseline_val_embeds.npz" `
    --output "output/baseline_results.json"
```

### Bước 2.2: Đánh giá mô hình sau khi Fine-tune
Sau khi hoàn tất quá trình huấn luyện ở Giai đoạn 1:

```powershell
python scripts/eval_youcookii.py `
    --adapter "D:\Học\KL\Code\Omni\Omni-fix\training\output\youcookii_ft" `
    --save-embeds "output/ft_val_embeds.npz" `
    --output "output/ft_results.json"
```

**Mẫu bảng kết quả hiển thị tự động:**
```text
=================================================================
               KẾT QUẢ ĐÁNH GIÁ YOUCOOKII (VAL)
=================================================================
Chỉ số          | Text -> Video (t2m)    | Video -> Text (m2t)   
-----------------------------------------------------------------
Recall@1        |              xx.xx%    |              xx.xx%   
Recall@5        |              xx.xx%    |              xx.xx%   
Recall@10       |              xx.xx%    |              xx.xx%   
MRR             |              xx.xx%    |              xx.xx%   
Median Rank     |                  x     |                  x    
=================================================================
```

---

## 3. Giai đoạn 3: Thực hiện truy vấn thực tế (Retrieval / Search)

Sau khi có mô hình, bạn có thể thực hiện tìm kiếm video theo câu mô tả bằng script [scripts/retrieval_demo.py](file:///d:/H%E1%BB%8Dc/KL/Code/Omni/Omni-fix/scripts/retrieval_demo.py).

### Cách 3.1: Tìm kiếm trực tiếp bằng câu lệnh CLI

```powershell
python scripts/retrieval_demo.py `
    --adapter "D:\Học\KL\Code\Omni\Omni-fix\training\output\youcookii_ft" `
    --gallery-embeds "output/ft_val_embeds.npz" `
    --query "spread margarine on two slices of white bread" `
    --top-k 5
```

**Kết quả trả về:**
```text
[Query] "spread margarine on two slices of white bread"

Hạng  | Score    | Video File                | Timestamps      | Ground Truth Caption
------------------------------------------------------------------------------------------
1     |  0.8421  | GLd3aX16zBg.mp4           | [90.0, 102.0]   | spread margarine on two slices of white bread
2     |  0.6912  | 3PzE0H46ZgI.mp4           | [45.0, 58.0]    | butter the bread slices
3     |  0.6350  | 8mNf90w0WbA.mp4           | [12.0, 20.0]    | apply butter on bread
...
```

### Cách 3.2: Chế độ tương tác nhập câu hỏi liên tục (`--interactive`)

```powershell
python scripts/retrieval_demo.py `
    --adapter "D:\Học\KL\Code\Omni\Omni-fix\training\output\youcookii_ft" `
    --gallery-embeds "output/ft_val_embeds.npz" `
    --interactive
```
Tại màn hình prompt, bạn chỉ cần gõ bất kỳ câu mô tả nấu ăn nào (ví dụ: `add cheese on bread`, `chop lettuce into a bowl`), hệ thống sẽ tính điểm tương đồng Cosine và trả về Top 5 video tương ứng ngay lập tức.

### Cách 3.3: Sử dụng trực tiếp bằng Python API

```python
from omniretriever import OmniRetriever

# 1. Khởi tạo mô hình
model = OmniRetriever.from_pretrained(
    base_model="D:/Học/KL/Code/Omni/WAVE_HOME/WAVE-7B",
    adapter="D:/Học/KL/Code/Omni/Omni-fix/training/output/youcookii_ft",
    device="cuda",
    dtype="bfloat16",
)

# 2. Mã hóa câu văn bản truy vấn ra vector 3584 chiều
z_text = model.encode_text("chop onions and garlic")

# 3. Mã hóa một event (hình ảnh + âm thanh) cắt từ video gốc theo metadata.
#    timestamps cắt frames; audio_path=video + audio_timestamps cắt audio của chính
#    video trên cùng cửa sổ -- đúng như lúc train.
video = "D:/Học/KL/Data/YouCookII/videos/GLd3aX16zBg.mp4"
event = (114.0, 127.0)
z_video = model.encode_av(video, timestamps=event, audio_path=video, audio_timestamps=event)

# 4. Tính độ tương đồng Cosine
score = float(z_text @ z_video.T)
print(f"Cosine Similarity: {score:.4f}")
```

---

## 4. Lưu ý quan trọng về phần cứng & Môi trường

1. **Yêu cầu GPU**:
   * Mô hình `WAVE-7B` có kích thước 7 tỷ tham số (~14 GB bfloat16).
   * Khi huấn luyện với LoRA + Gradient Checkpointing + Batch Size 1, cần GPU có tối thiểu **16 GB – 24 GB VRAM** (RTX 3090 / 4090, A5000, L4, A100).
   * Không nên train trên CPU vì sẽ bị quá tải RAM (OOM) hoặc chạy rất lâu.

2. **Cài đặt PyTorch CUDA (nếu máy có GPU NVIDIA)**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Xử lý tiếng Việt trên Windows PowerShell**:
   Luôn chạy lệnh sau trước mỗi phiên làm việc:
   ```powershell
   $env:PYTHONIOENCODING = "utf-8"
   ```
