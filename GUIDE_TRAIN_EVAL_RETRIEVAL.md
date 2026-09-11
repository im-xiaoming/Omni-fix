# HƯỚNG DẪN CHI TIẾT: FINE-TUNE, EVAL VÀ RETRIEVAL (OMNIRETRIEVER-7B)

Tài liệu này hướng dẫn toàn bộ quy trình thực hiện với mô hình **OmniRetriever-7B** trên bộ dữ liệu **YouCookII**:
1. **Fine-tuning**: Huấn luyện thích nghi miền từ adapter có sẵn.
2. **Evaluation**: Đánh giá các chỉ số truy vấn (Recall@1/5/10, MRR, Median Rank) trên tập Validation (`val_omni.jsonl`).
3. **Retrieval**: Thực hiện tìm kiếm video từ văn bản (Text $\rightarrow$ Video) hoặc ngược lại.

---

## 0. Bảng cấu hình đường dẫn chuẩn

Toàn bộ script trong repository đã được cấu hình sẵn các đường dẫn mặc định sau:

| Tên biến | Đường dẫn trên máy | Mô tả |
| :--- | :--- | :--- |
| **`WAVE_PATH`** | `D:\Học\KL\Code\Omni\WAVE_HOME\WAVE-7B` | Trọng số mô hình nền WAVE-7B |
| **`BEATS_PATH`** | `D:\Học\KL\Code\Omni\WAVE_HOME\BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` | Checkpoint Audio Encoder BEATs |
| **`DATA_PATH`** | `D:\Học\KL\Data\YouCookII\YouCookII\metadata\train_omni.jsonl` | Tập metadata huấn luyện (8,816 mẫu) |
| **`VAL_MANIFEST`** | `D:\Học\KL\Data\YouCookII\YouCookII\metadata\val_omni.jsonl` | Tập metadata validation (3,110 mẫu) |
| **`VIDEO_ROOT`** | `D:\Học\KL\Data\YouCookII\YouCookII\videos` | Thư mục chứa các file video `.mp4` |
| **`AUDIO_ROOT`** | `D:\Học\KL\Data\YouCookII\YouCookII\audio` | Thư mục chứa các file audio `.wav` |
| **`LORA_CKPT`** | `D:\Học\KL\Code\Omni\adapters\omniretriever-7b` | Adapter LoRA gốc của OmniRetriever |
| **`OUTPUT_DIR`** | `D:\Học\KL\Code\Omni\Omni-fix\training\output\omniretriever_7b` | Thư mục lưu checkpoint sau khi train |

---

## 1. Giai đoạn 1: Huấn luyện (Fine-tuning)

### Cơ chế đóng băng mô hình (Freezing & Unfreezing)
* **Đóng băng 100% Backbone**: Toàn bộ 7 tỷ tham số của LLM Qwen2.5, Vision Tower (ViT), Audio Tower (Whisper) và BEATs Transformer đều được khóa cứng (`requires_grad = False`).
* **Chỉ mở các lớp thích nghi**:
  1. Các ma trận LoRA ($r=16, \alpha=32$) ở các lớp Attention `(q|k|v)_proj`.
  2. Lớp phân loại chiếu đa phương thức `model.classify_linear` (chiếu sang 3584-dim).
  3. Lớp chiếu âm thanh BEATs `model.beats_proj` và `model.beats_ln`.
* Tổng số tham số huấn luyện chỉ chiếm **~0.5% - 1%**, giúp tiết kiệm VRAM và giữ nguyên tri thức gốc.

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

export VIDEO_ROOT="D:/Học/KL/Data/YouCookII/YouCookII/videos"
export AUDIO_ROOT="D:/Học/KL/Data/YouCookII/YouCookII/audio"
export WAVE_PATH="D:/Học/KL/Code/Omni/WAVE_HOME/WAVE-7B"
export BEATS_PATH="D:/Học/KL/Code/Omni/WAVE_HOME/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
export DATA_PATH="D:/Học/KL/Data/YouCookII/YouCookII/metadata/train_omni.jsonl"
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

## 2. Giai đoạn 2: Đánh giá mô hình (Evaluation)

Trong bài toán Video/Audio-Text Retrieval, việc đánh giá không dùng cross-entropy loss thông thường mà đo trực tiếp bằng các chỉ số xếp hạng tìm kiếm trên tập Validation (`val_omni.jsonl`):
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

# Hoặc chạy toàn bộ 3,110 mẫu validation và lưu embedding:
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

# 3. Mã hóa video clip (hình ảnh + âm thanh)
z_video = model.encode_av("D:/Học/KL/Data/YouCookII/YouCookII/videos/sample.mp4")

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
