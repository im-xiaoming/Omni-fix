# HƯỚNG DẪN CHẠY ĐÁNH GIÁ BENCHMARK (12 HƯỚNG TRUY VẤN)

Tài liệu này hướng dẫn cách lấy mô hình đã fine-tune và chấm điểm nó trên benchmark
theo **12 hướng truy vấn** của OmniRetriever.

Metadata benchmark dùng **đúng định dạng như metadata training** (`train_omni.jsonl` /
`val_omni.jsonl`): tên file để trần + `timestamps`. Pipeline inference đã được cập nhật
để đọc thẳng định dạng này, không cần chuyển đổi gì.

Quy trình gồm 2 lệnh: **trích xuất embedding** → **chấm điểm**.

---

## 0. Chuẩn bị (chỉ làm 1 lần)

### Cài thư viện

```bash
pip install -r requirements.txt
```

### Vá mô hình WAVE-7B

```bash
python scripts/patch_wave_rope.py /đường/dẫn/WAVE-7B
```

Lệnh này sửa một lỗi tính vị trí M-RoPE trong code của WAVE-7B. **Bắt buộc** nếu bạn chấm
các hướng có dùng `av` (`t2av`, `av2t`, `v2at`, `at2v`); không vá thì nhánh `av` sẽ báo lỗi
shape mismatch. Script tự lưu bản gốc thành `.prepatch`, và chạy lại lần hai sẽ tự bỏ qua.

### Đường dẫn cần có

| Biến | Ví dụ | Mô tả |
| :--- | :--- | :--- |
| `WAVE_PATH` | `D:\Học\KL\Code\Omni\WAVE_HOME\WAVE-7B` | Mô hình nền WAVE-7B (đã vá ở trên) |
| `ADAPTER` | `training/output/omniretriever_7b/checkpoint-2000` | **Adapter bạn vừa fine-tune** |
| `MANIFEST` | `D:\Học\KL\Data\YouCookII\YouCookII\metadata\val_omni.jsonl` | Metadata benchmark |
| `VIDEO_ROOT` | `D:\Học\KL\Data\YouCookII\YouCookII\videos` | Thư mục chứa `.mp4` |
| `AUDIO_ROOT` | `D:\Học\KL\Data\YouCookII\YouCookII\audio` | Thư mục chứa `.wav` |

`ADAPTER` là thư mục có `adapter_config.json` + `adapter_model.safetensors`. Trỏ vào
`checkpoint-XXXX` của bạn để chấm mô hình mình, hoặc vào `adapters/omniretriever-7b`
để chấm adapter gốc của tác giả làm mốc so sánh.

---

## 1. Trích xuất embedding

```bash
export PYTHONPATH=src
export VIDEO_ROOT="D:/Học/KL/Data/YouCookII/YouCookII/videos"
export AUDIO_ROOT="D:/Học/KL/Data/YouCookII/YouCookII/audio"

python -m omniretriever.cli extract \
    "D:/Học/KL/Data/YouCookII/YouCookII/metadata/val_omni.jsonl" \
    --base-model "D:/Học/KL/Code/Omni/WAVE_HOME/WAVE-7B" \
    --adapter    "training/output/omniretriever_7b/checkpoint-2000" \
    --output     "output/embeddings.npz" \
    --device cuda --dtype bfloat16 -v
```

Trên Windows PowerShell thì đặt biến môi trường bằng `$env:VIDEO_ROOT = "..."`.

Mỗi record sinh ra tối đa 6 embedding (`text`, `video`, `audio`, `av`, `tv`, `at`), đủ để
chấm cả 12 hướng. Kết quả ghi vào một file `.npz`.

**Chạy lại khi bị ngắt**: cứ 25 batch, tiến độ được lưu vào `output/embeddings.partial.npz`.
Chạy lại đúng lệnh trên là nó tiếp tục từ chỗ dừng. Muốn làm lại từ đầu thì thêm `--no-resume`.

---

## 2. Chấm điểm

```bash
python scripts/score_embeddings.py output/embeddings.npz --output output/metrics.json
```

Kết quả in ra dạng:

```
huong        R@1      R@5     R@10      MRR  gallery
t2v       0.8333   0.9400   0.9667   0.8832     3031
v2t       0.8733   0.9467   0.9700   0.9049     3031
...
AVG-single  R@1: 0.4906   (6 huong)
AVG-dual    R@1: 0.6883   (6 huong)
AVG-all     R@1: 0.5894   (12 huong)
```

Cột `gallery` là số record được chấm ở hướng đó. **Recall phụ thuộc cỡ gallery** — gallery
càng nhỏ thì điểm càng cao — nên chỉ so sánh hai con số khi cỡ gallery bằng nhau. Khi so với
số liệu trong bài báo, nhớ kiểm tra cỡ gallery của họ trước.

---

## 3. Chạy nhanh để thử

Trước khi chạy nguyên tập, nên thử trên vài trăm record:

```bash
head -300 val_omni.jsonl > val300.jsonl
```

Rồi chạy 2 lệnh trên với `val300.jsonl`. Nếu chỉ muốn xem 2 hướng text–video cho nhanh:

```bash
python -m omniretriever.cli extract val300.jsonl ... --modalities text video
```

`--modalities` là cách duy nhất thật sự rút ngắn thời gian chạy, vì mỗi modality là một
lượt quét riêng qua manifest. Bỏ `tv` và `at` cắt được 1/3 khối lượng, đổi lại mất 4 hướng
`a2tv`, `tv2a`, `v2at`, `at2v`.

---

## 4. Vài điểm cần biết

**`timestamps` được xử lý tự động.** Trong YouCookII, tới 15 record cùng dùng một file
video và chỉ khác nhau ở `timestamps`. Pipeline chỉ decode đúng đoạn đó. Nếu bỏ qua
`timestamps`, cả 15 segment sẽ ra embedding **giống hệt nhau** và 8/12 hướng chấm sẽ vô nghĩa.
Đây cũng là lý do nó chạy nhanh: decode cả video mất ~89 giây, decode đúng segment chỉ ~1–3 giây.

**Audio lấy từ đâu.** Hướng `audio` và `at` đọc file `.wav` trong `AUDIO_ROOT` (vốn đã được
cắt sẵn theo từng segment). Riêng hướng `av` đọc **cả hình lẫn tiếng từ chính file video**,
vì `use_audio_in_video` cần hai luồng khớp thời gian với nhau.

**Giữ `--batch-size 1`.** Tăng lên sẽ làm đổi embedding của mọi mẫu không phải mẫu dài nhất
trong batch, do lớp fusion lấy vector ở vị trí token cuối cố định còn phần padding thì không
được bù trừ.

**Đừng trộn embedding cũ và mới.** Prompt hiện đã thêm token `<|im_end|>` ở cuối cho khớp
lúc training, nên các file `.npz` trích trước đây không còn so sánh được với file mới. Muốn
so sánh thì trích lại toàn bộ.

**Điều kiện phần cứng.** Cần GPU; tham chiếu là A100-40GB với `--dtype bfloat16`. Máy chỉ có
CPU sẽ không nạp nổi mô hình 7B.
