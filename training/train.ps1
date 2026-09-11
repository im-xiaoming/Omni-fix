# PowerShell training launcher for OmniRetriever-7B on Windows
#
# Usage:
#   .\training\train.ps1
#
# Hoặc truyền tham số tùy chỉnh:
#   .\training\train.ps1 -BatchSize 2 -Epochs 3

[CmdletBinding()]
param (
    [string]$WavePath = $env:WAVE_PATH,
    [string]$BeatsPath = $env:BEATS_PATH,
    [string]$DataPath = $env:DATA_PATH,
    [string]$VideoRoot = $env:VIDEO_ROOT,
    [string]$AudioRoot = $env:AUDIO_ROOT,
    [string]$ImageRoot = $env:IMAGE_ROOT,
    [string]$OutputDir = $env:OUTPUT_DIR,
    [int]$NumGpus = 1,
    [int]$MasterPort = 29503,
    [int]$Epochs = 1,
    [int]$BatchSize = 1,
    [int]$GradAccum = 8,
    [double]$Lr = 1e-5,
    [int]$LoraR = 16,
    [int]$LoraAlpha = 32,
    [int]$DataLoaderWorkers = 0,
    [string]$LoraCkpt = $env:LORA_CKPT,
    [switch]$LoraInitOnly,
    [bool]$GradientCheckpointing = $true,
    [string]$UseTupleInfonce = "True",
    [switch]$UseDeepspeed,
    [switch]$Bf16,
    [switch]$Fp16,
    [switch]$DryRun,
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

if ($Help -or ($WavePath -eq "--help") -or ($WavePath -eq "-h")) {
    Write-Host @"
Cach su dung train.ps1:
  .\training\train.ps1                                   # Chay voi thong so mac dinh
  .\training\train.ps1 -DryRun                           # Kiem tra path va cau hinh (khong chay train)
  .\training\train.ps1 -LoraCkpt "..\..\adapters\omniretriever-7b" -LoraInitOnly   # Fine-tune tiep tu adapter co san
  .\training\train.ps1 -BatchSize 1 -GradAccum 8 -GradientCheckpointing            # Tiet kiem VRAM GPU
  .\training\train.ps1 -OutputDir "D:\output\test"       # Thu muc luu checkpoint

Cac bien moi truong tu dong nhan dien:
  VIDEO_ROOT, AUDIO_ROOT, WAVE_PATH, BEATS_PATH, DATA_PATH, LORA_CKPT
"@
    exit 0
}

# Thiết lập encoding UTF-8 để hỗ trợ đường dẫn tiếng Việt (ví dụ: 'D:\Học\KL')
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Thiết lập giá trị mặc định cho các đường dẫn nếu chưa được đặt
# Sử dụng đường dẫn tương đối từ vị trí script để tránh lỗi encoding tiếng Việt
if (-not $WavePath) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\WAVE_HOME\WAVE-7B"))
    if (Test-Path $candidate) { $WavePath = $candidate } else { $WavePath = "D:\Học\KL\Code\Omni\WAVE_HOME\WAVE-7B" }
}
if (-not $BeatsPath) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\WAVE_HOME\BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"))
    if (Test-Path $candidate) { $BeatsPath = $candidate } else { $BeatsPath = "D:\Học\KL\Code\Omni\WAVE_HOME\BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" }
}
if (-not $DataPath) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\..\..\Data\YouCookII\YouCookII\metadata\train_omni.jsonl"))
    if (Test-Path $candidate) { $DataPath = $candidate } else { $DataPath = "D:\Học\KL\Data\YouCookII\YouCookII\metadata\train_omni.jsonl" }
}
if (-not $VideoRoot) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\..\..\Data\YouCookII\YouCookII\videos"))
    if (Test-Path $candidate) { $VideoRoot = $candidate } else { $VideoRoot = "D:\Học\KL\Data\YouCookII\YouCookII\videos" }
}
if (-not $AudioRoot) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\..\..\Data\YouCookII\YouCookII\audio"))
    if (Test-Path $candidate) { $AudioRoot = $candidate } else { $AudioRoot = "D:\Học\KL\Data\YouCookII\YouCookII\audio" }
}
if (-not $LoraCkpt) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\adapters\omniretriever-7b"))
    if (Test-Path $candidate) { $LoraCkpt = $candidate } else { $LoraCkpt = "D:\Học\KL\Code\Omni\adapters\omniretriever-7b" }
}
if (-not $OutputDir) {
    $OutputDir = "$PSScriptRoot\output\omniretriever_7b"
}

# Export các biến môi trường cho Data Loader và Model
$env:WAVE_PATH = $WavePath
$env:BEATS_PATH = $BeatsPath
$env:DATA_PATH = $DataPath
$env:VIDEO_ROOT = $VideoRoot
$env:AUDIO_ROOT = $AudioRoot
if ($LoraCkpt -and ($LoraCkpt -ne "No")) { $env:LORA_CKPT = $LoraCkpt }
if ($ImageRoot) { $env:IMAGE_ROOT = $ImageRoot }

$RepoRoot = $PSScriptRoot
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$RepoRoot;$($env:PYTHONPATH)"
} else {
    $env:PYTHONPATH = "$RepoRoot"
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  OmniRetriever-7B Training Launcher (Windows PowerShell)   " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "WAVE_PATH  : $WavePath"
Write-Host "BEATS_PATH : $BeatsPath"
Write-Host "DATA_PATH  : $DataPath"
Write-Host "VIDEO_ROOT : $VideoRoot"
Write-Host "AUDIO_ROOT : $AudioRoot"
Write-Host "LORA_CKPT  : $LoraCkpt"
Write-Host "OUTPUT_DIR : $OutputDir"
Write-Host "BATCH_SIZE : $BatchSize | GRAD_ACCUM: $GradAccum | EPOCHS: $Epochs"
Write-Host "============================================================" -ForegroundColor Cyan

# Kiểm tra đường dẫn tồn tại
if (-not (Test-Path $WavePath)) {
    Write-Warning "Không tìm thấy thư mục WAVE_PATH: $WavePath"
}
if (-not (Test-Path $BeatsPath)) {
    Write-Warning "Không tìm thấy file BEATS_PATH: $BeatsPath"
}
if (-not (Test-Path $DataPath)) {
    Write-Warning "Không tìm thấy file DATA_PATH: $DataPath"
}
if ($LoraCkpt -and ($LoraCkpt -ne "No") -and (-not (Test-Path $LoraCkpt))) {
    Write-Warning "Không tìm thấy thư mục LORA_CKPT: $LoraCkpt"
}

# Kiểm tra GPU / CUDA
$hasCuda = $false
try {
    $cudaCheck = python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cudaCheck -like "*True*") {
        $hasCuda = $true
        $gpuName = python -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
        Write-Host "[OK] Tìm thấy GPU CUDA: $gpuName" -ForegroundColor Green
    }
} catch {}

if (-not $hasCuda) {

    Write-Warning "IMPORTANT NOTICE: The current Python environment does not have an NVIDIA CUDA GPU (CPU only)."
    Write-Warning "The WAVE-7B model has 7 billion parameters (~14 GB of weights), so training"
    Write-Warning "it on a personal computer using CPU may easily cause an out-of-memory (OOM) error"
    Write-Warning "or be extremely slow."
    Write-Warning "For practical fine-tuning, you should use a machine with an NVIDIA GPU"
    Write-Warning "(>= 16 GB-24 GB VRAM) or use a Cloud GPU / WSL2."

}

# Tạo thư mục output nếu chưa có
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Thiết lập precision
$precisionArgs = @()
if ($Bf16) {
    $precisionArgs += @("--bf16", "True")
} elseif ($Fp16) {
    $precisionArgs += @("--fp16", "True")
} elseif ($hasCuda) {
    # Mặc định dùng bf16 nếu có CUDA
    $precisionArgs += @("--bf16", "True")
}

# Tập hợp các tham số huấn luyện
$trainArgs = @(
    "$RepoRoot\qwenvl\train\train_qwen.py",
    "--model_name_or_path", "$WavePath",
    "--model_base", "$WavePath",
    "--dataset_use", "$DataPath",
    "--output_dir", "$OutputDir",
    "--num_train_epochs", "$Epochs",
    "--per_device_train_batch_size", "$BatchSize",
    "--gradient_accumulation_steps", "$GradAccum",
    "--learning_rate", "$Lr",
    "--weight_decay", "0.01",
    "--warmup_ratio", "0.03",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "1",
    "--model_max_length", "2048",
    "--dataloader_num_workers", "$DataLoaderWorkers",
    "--train_classify", "True",
    "--classify_type", "all_layer",
    "--pred_embeds", "True",
    "--use_beats", "True",
    "--tune_beats_proj", "True",
    "--fixed_audio_duration", "8",
    "--video_max_frames", "8",
    "--video_min_frames", "8",
    "--max_pixels", "50176",
    "--min_pixels", "50176",
    "--use_lora", "True",
    "--lora_r", "$LoraR",
    "--lora_alpha", "$LoraAlpha",
    "--use_tuple_infonce", "$UseTupleInfonce",
    "--save_strategy", "steps",
    "--save_steps", "1000",
    "--save_total_limit", "5",
    "--report_to", "none"
) + $precisionArgs

if ($LoraCkpt -and ($LoraCkpt -ne "No")) {
    $trainArgs += @("--lora_ckpt", "$LoraCkpt", "--lora_init_only", "True")
}
if ($GradientCheckpointing) {
    $trainArgs += @("--gradient_checkpointing", "True")
}

if ($ExtraArgs) {
    $trainArgs += $ExtraArgs
}

if ($DryRun) {
    Write-Host "[DryRun] Kiểm tra thành công! Lệnh sẽ được thực thi như sau:" -ForegroundColor Green
    Write-Host "python $($trainArgs -join ' ')"
    exit 0
}

# Kiểm tra DeepSpeed
$dsInstalled = Get-Command deepspeed -ErrorAction SilentlyContinue

if ($UseDeepspeed -and $dsInstalled) {
    Write-Host "Chạy với DeepSpeed (GPUs: $NumGpus, Port: $MasterPort)..." -ForegroundColor Yellow
    $dsConfig = "$RepoRoot\configs\ds_zero0.json"
    deepspeed --num_gpus=$NumGpus --master_port=$MasterPort @trainArgs --deepspeed "$dsConfig"
} else {
    Write-Host "Chạy trực tiếp với Python..." -ForegroundColor Yellow
    python @trainArgs
}
