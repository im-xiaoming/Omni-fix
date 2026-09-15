# HƯỚNG DẪN CHẠY ĐÁNH GIÁ BENCHMARK (12 HƯỚNG TRUY VẤN)

Tài liệu này hướng dẫn cách lấy mô hình đã fine-tune và chấm điểm nó trên benchmark
OmniRetriever theo **12 hướng truy vấn**.

Quy trình gồm 2 lệnh: **trích xuất embedding** → **chấm điểm**.

---

## 0. Dữ liệu benchmark

Benchmark nằm ở `D:\Học\KL\Data\benchmark`:

```
benchmark/
├── manifests/
│   └── omniretriever_bench_final.jsonl     3510 record
└── videos/
    ├── 6886118185460845830.mp4             hình
    ├── 6886118185460845830.wav             tiếng (16 kHz, mono)
    └── ...
```

Mỗi record trong manifest có đúng 4 trường:

```json
{"text": "Both women extend their arms horizontally...",
 "video": "benchmark/videos/6886118185460845830.mp4",
 "audio": "benchmark/videos/6886118185460845830.wav",
 "id": "00000"}
```

Hai điểm khác với metadata training, và cả hai đều đã được pipeline xử lý sẵn:

**Clip đã cắt sẵn, không có `timestamps`.** Mỗi `.mp4` là một segment hoàn chỉnh, và
`.wav` đi kèm khớp đúng khoảng thời gian đó. Nên ở đây không có bước cắt theo segment
như lúc training — pipeline đọc trọn file. (Code cắt theo `timestamps` vẫn còn để dùng
cho manifest kiểu YouCookII; xem mục 4.)

**Đường dẫn là tương đối** (`benchmark/videos/...`), đã kèm sẵn thư mục. Dùng
`--media-root` trỏ vào thư mục **chứa** `benchmark` để nối vào:

| Chạy ở đâu | `--media-root` |
| :--- | :--- |
| Máy local | `D:/Học/KL/Data` |
| Colab (theo `Omni.ipynb`) | `/content` |

---

## 1. Chuẩn bị (chỉ làm 1 lần)

### Cài thư viện

```bash
pip install -r requirements.txt
```

### Vá mô hình WAVE-7B

```bash
python scripts/patch_wave_rope.py /đường/dẫn/WAVE_HOME/WAVE-7B
```

Lệnh này sửa một lỗi tính vị trí M-RoPE trong code của WAVE-7B. **Bắt buộc** nếu bạn chấm
các hướng có dùng `av` (`t2av`, `av2t`, `v2at`, `at2v`); không vá thì nhánh `av` sẽ báo lỗi
shape mismatch. Script tự lưu bản gốc thành `.prepatch`, và chạy lại lần hai sẽ tự bỏ qua.

### Đường dẫn cần có

| Biến | Ví dụ | Mô tả |
| :--- | :--- | :--- |
| `BASE_MODEL` | `D:\Học\KL\Code\Omni\WAVE_HOME\WAVE-7B` | Mô hình nền WAVE-7B (đã vá ở trên) |
| `ADAPTER` | `training/output/omniretriever_7b/checkpoint-1102` | **Adapter bạn vừa fine-tune** |
| `MANIFEST` | `D:\Học\KL\Data\benchmark\manifests\omniretriever_bench_final.jsonl` | Metadata benchmark |
| `MEDIA_ROOT` | `D:\Học\KL\Data` | Thư mục **chứa** `benchmark/` |

`ADAPTER` là thư mục có `adapter_config.json` + `adapter_model.safetensors`. Trỏ vào
`checkpoint-XXXX` của bạn để chấm mô hình mình, hoặc vào `adapters/omniretriever-7b`
để chấm adapter gốc của tác giả làm mốc so sánh.

---

## 2. Trích xuất embedding

```bash
export PYTHONPATH=src

python -m omniretriever.cli extract \
    "D:/Học/KL/Data/benchmark/manifests/omniretriever_bench_final.jsonl" \
    --media-root "D:/Học/KL/Data" \
    --base-model "D:/Học/KL/Code/Omni/WAVE_HOME/WAVE-7B" \
    --adapter    "training/output/omniretriever_7b/checkpoint-1102" \
    --output     "output/embeddings.npz" \
    --device cuda --dtype bfloat16 -v
```

Trên Windows PowerShell thì đặt biến môi trường bằng `$env:PYTHONPATH = "src"`.

Mỗi record sinh ra 6 embedding (`text`, `video`, `audio`, `av`, `tv`, `at`), đủ để chấm
cả 12 hướng. Kết quả ghi vào một file `.npz`.

**Kiểm tra đường dẫn ngay từ đầu.** Nếu `--media-root` sai, lệnh in cảnh báo kiểu
`N of 7020 media files do not exist ...` ngay trong giây đầu, kèm 3 đường dẫn hỏng đầu
tiên. Sửa rồi chạy lại, đừng để nó chạy tiếp — gặp file thiếu là cả lượt chạy dừng.

**Chạy lại khi bị ngắt**: cứ 25 batch, tiến độ được lưu vào `output/embeddings.partial.npz`.
Chạy lại đúng lệnh trên là nó tiếp tục từ chỗ dừng. Muốn làm lại từ đầu thì thêm `--no-resume`.

---

## 3. Chấm điểm

```bash
python scripts/score_embeddings.py output/embeddings.npz --output output/metrics.json
```

Kết quả in ra dạng:

```
huong        R@1      R@5     R@10      MRR  gallery
t2v       0.8333   0.9400   0.9667   0.8832     3510
v2t       0.8733   0.9467   0.9700   0.9049     3510
...
AVG-single  R@1: 0.4906   (6 huong)
AVG-dual    R@1: 0.6883   (6 huong)
AVG-all     R@1: 0.5894   (12 huong)
```

Cột `gallery` là số record được chấm ở hướng đó. **Recall phụ thuộc cỡ gallery** — gallery
càng nhỏ thì điểm càng cao — nên chỉ so sánh hai con số khi cỡ gallery bằng nhau. Benchmark
này có 3510 record, còn bài báo chấm trên 3782, nên điểm ở đây sẽ hơi cao hơn ở cùng một
chất lượng mô hình.

---

## 4. Chạy nhanh để thử

Trước khi chạy nguyên 3510 record, nên thử trên vài trăm:

```bash
head -300 "D:/Học/KL/Data/benchmark/manifests/omniretriever_bench_final.jsonl" > bench300.jsonl
```

Rồi chạy 2 lệnh trên với `bench300.jsonl`. Nếu chỉ muốn xem 2 hướng text–video cho nhanh:

```bash
python -m omniretriever.cli extract bench300.jsonl ... --modalities text video
```

`--modalities` là cách duy nhất thật sự rút ngắn thời gian chạy, vì mỗi modality là một
lượt quét riêng qua manifest. Bỏ `tv` và `at` cắt được 1/3 khối lượng, đổi lại mất 4 hướng
`a2tv`, `tv2a`, `v2at`, `at2v`.

---

## 5. Vài điểm cần biết

**Audio lấy từ đâu.** Cả 3 hướng có tiếng — `audio`, `at` và `av` — đều đọc file `.wav`
trong trường `audio` của record. Trước đây riêng `av` đọc tiếng từ chính file `.mp4`;
giờ đã sửa cho khớp với lúc training, vì `training/qwenvl/data/data_qwen.py` cũng nạp
`use_audio_in_video` từ trường `audio` chứ không phải từ container video. Sửa chỗ này
còn làm cho `audio` và `av` dùng **đúng cùng một waveform**, nên các hướng so hai bên với
nhau mới có nghĩa.

**Không cần cắt segment ở benchmark.** Benchmark không có `timestamps` nên mọi file được
đọc trọn. Cơ chế cắt vẫn còn cho manifest kiểu YouCookII, nơi tới 15 record dùng chung một
file video và chỉ khác `timestamps`; ở đó nếu bỏ qua `timestamps` thì cả 15 segment sẽ ra
embedding **giống hệt nhau**. Với manifest kiểu đó, dùng `$VIDEO_ROOT` / `$AUDIO_ROOT`
thay cho `--media-root`, vì tên file ở đó để trần không kèm thư mục.

**Giữ `--batch-size 1`.** Tăng lên sẽ làm đổi embedding của mọi mẫu không phải mẫu dài nhất
trong batch, do lớp fusion lấy vector ở vị trí token cuối cố định còn phần padding thì không
được bù trừ.

**Đừng trộn embedding cũ và mới.** Hai thay đổi khiến file `.npz` trích trước đây không còn
so sánh được với file mới: prompt đã thêm token `<|im_end|>` ở cuối cho khớp lúc training,
và `av` đã đổi nguồn audio. Muốn so sánh thì trích lại toàn bộ.

**Điều kiện phần cứng.** Cần GPU; tham chiếu là A100-40GB với `--dtype bfloat16`. Máy chỉ có
CPU sẽ không nạp nổi mô hình 7B.
