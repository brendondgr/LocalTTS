# Install kokoro-tts on Windows into one self-contained environment and put a `kokoro-tts`
# launcher on PATH. (Linux/macOS: tts/install.sh.)
#
#   powershell -ExecutionPolicy Bypass -File tts\install.ps1                 # auto-detect GPU
#   powershell -ExecutionPolicy Bypass -File tts\install.ps1 -Gpu cuda -Cuda cu130
#   powershell -ExecutionPolicy Bypass -File tts\install.ps1 -Gpu rocm -RocmArch gfx1151
#   powershell -ExecutionPolicy Bypass -File tts\install.ps1 -Gpu cpu
#
# Needs uv (winget install astral-sh.uv) and, recommended, eSpeak NG (winget install eSpeak-NG.eSpeak-NG).
# NVIDIA: any recent driver. AMD: ROCm-on-Windows wheels from AMD's TheRock nightlies exist for
# Strix Halo (gfx1151) and other RDNA3/4 families; they are previews, so -Gpu cpu is the fallback.
# NOTE: written to mirror install.sh; not yet exercised on a Windows machine.
param(
  [ValidateSet('auto', 'cuda', 'rocm', 'cpu')] [string]$Gpu = 'auto',
  [string]$RocmArch = '',
  [string]$RocmIndex = '',
  [string]$Cuda = 'cu130',
  [string]$Prefix = "$HOME\venvs\kokoro",
  [string]$Bin = "$HOME\.local\bin",
  [switch]$ForceTorch
)
$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
function Say($m) { Write-Host "» $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "! $m" -ForegroundColor Yellow }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv not found. Install it: winget install astral-sh.uv   (then open a new terminal)' }
if (-not (Get-Command espeak-ng -ErrorAction SilentlyContinue)) { Warn 'eSpeak NG not on PATH (Kokoro bundles a copy; the system one is the reliable fallback): winget install eSpeak-NG.eSpeak-NG' }

# ---- pick the backend --------------------------------------------------------------------------
if ($Gpu -eq 'auto') {
  $names = (Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }) -join ' | '
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $Gpu = 'cuda' }
  elseif ($names -match 'Radeon') { $Gpu = 'rocm' }
  else { $Gpu = 'cpu' }
  Say "GPU(s): $names → backend $Gpu"
}
if ($Gpu -eq 'rocm' -and -not $RocmIndex) {
  if (-not $RocmArch) {
    $names = (Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }) -join ' '
    if ($names -match '8060S|8050S|8040S|AI Max') { $RocmArch = 'gfx1151' }
    elseif ($names -match '890M|880M') { $RocmArch = 'gfx1150' }
    elseif ($names -match 'RX 9\d{3}') { $RocmArch = 'gfx120X-all' }
    elseif ($names -match 'RX 7\d{3}|780M|760M') { $RocmArch = 'gfx110X-all' }
    else { throw "Could not map '$names' to a ROCm target. Pass -RocmArch (e.g. gfx1151, gfx110X-all, gfx120X-all) or use -Gpu cpu." }
  }
  $RocmIndex = "https://rocm.nightlies.amd.com/v2/$RocmArch/"
  Say "AMD $RocmArch → $RocmIndex"
}
switch ($Gpu) {
  'cuda' { $TorchArgs = @('--index-url', "https://download.pytorch.org/whl/$Cuda", 'torch', 'torchaudio') }
  'rocm' { $TorchArgs = @('--pre', '--index-url', $RocmIndex, 'torch', 'torchaudio') }
  'cpu'  { $TorchArgs = @('--index-url', 'https://download.pytorch.org/whl/cpu', 'torch', 'torchaudio') }
}

# ---- environment -------------------------------------------------------------------------------
$Py = Join-Path $Prefix 'Scripts\python.exe'
if (-not (Test-Path $Py)) { Say "creating Python 3.12 environment at $Prefix"; uv venv $Prefix --python 3.12 }
$env:VIRTUAL_ENV = $Prefix

function BackendOk {
  $code = "import torch,sys; w='$Gpu'; sys.exit(0 if {'rocm':bool(torch.version.hip),'cuda':bool(torch.version.cuda) and not torch.version.hip,'cpu':True}[w] else 1)"
  & $Py -c $code 2>$null; return ($LASTEXITCODE -eq 0)
}
# Torch first: installing kokoro first would pull the default torch from PyPI.
if ($ForceTorch -or -not (BackendOk)) {
  Say "installing torch ($Gpu)"
  if ($ForceTorch) { uv pip install --reinstall @TorchArgs } else { uv pip install @TorchArgs }
} else { Say "torch already matches backend $Gpu — keeping it" }

Say 'installing kokoro, soundfile and the spaCy English model'
uv pip install 'kokoro>=0.9.4' soundfile 'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
if (-not (BackendOk)) { Warn "kokoro's install replaced the $Gpu torch — reinstalling it"; uv pip install --reinstall @TorchArgs }

# ---- program + launcher --------------------------------------------------------------------------
Copy-Item (Join-Path $Here 'kokoro_tts.py') (Join-Path $Prefix 'kokoro_tts.py') -Force
New-Item -ItemType Directory -Force -Path $Bin | Out-Null
$launcher = Join-Path $Bin 'kokoro-tts.cmd'
@"
@echo off
rem kokoro-tts launcher (installed by video-maker\tts\install.ps1). Environment: $Prefix
if "%KOKORO_HOME%"=="" set "KOKORO_HOME=$Prefix"
"%KOKORO_HOME%\Scripts\python.exe" "%KOKORO_HOME%\kokoro_tts.py" %*
"@ | Set-Content -Encoding ASCII $launcher

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (($userPath -split ';') -notcontains $Bin) {
  [Environment]::SetEnvironmentVariable('Path', "$userPath;$Bin", 'User')
  Warn "added $Bin to your user PATH — open a new terminal to pick it up"
  $env:Path = "$env:Path;$Bin"
}
Say 'smoke test (first run downloads the model, ~330 MB, and compiles GPU kernels)'
& $launcher --check
Say 'done. Try:  kokoro-tts "Hello from Kokoro." -o hello.wav'
