# ATLAS installer (Windows 10/11). Run from the ATLAS folder:
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# Everything is local and account-free: no API keys, no sign-ins. What it does:
#   1. Python 3.12 + Node.js LTS (via winget, only if missing)
#   2. A private virtual environment in .venv (never touches system Python)
#   3. PyTorch (CUDA build; an NVIDIA GPU is required for the voice engine)
#   4. ATLAS's Python packages, the desktop shell, and the speech models
#   5. A desktop + Start menu shortcut
# Re-running is safe: finished steps are skipped. Choosing a language model
# (local via Ollama, or a cloud provider with your own key) happens in the
# first-run setup inside the app.
param([switch]$Cpu, [switch]$SkipModels, [switch]$NoShortcut)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Log = Join-Path $Root 'install.log'
try { "=== ATLAS install $(Get-Date -Format s) ===" | Out-File $Log -Encoding utf8 }
catch {  # log held open by an earlier, still-running install: use a fresh file
    $Log = Join-Path $Root ("install-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    "=== ATLAS install $(Get-Date -Format s) ===" | Out-File $Log -Encoding utf8
}

function Step($n, $total, $msg) { Write-Host ""; Write-Host "[$n/$total] $msg" -ForegroundColor Cyan; "[$n/$total] $msg" | Out-File $Log -Append -Encoding utf8 }
function Ok($msg)   { Write-Host "  ok  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  !!  $msg" -ForegroundColor Yellow; "WARN $msg" | Out-File $Log -Append -Encoding utf8 }
function Fail($msg) {
    Write-Host ""; Write-Host "  Install stopped: $msg" -ForegroundColor Red
    Write-Host "  Details are in $Log. Fix the problem above and run install.ps1 again; finished steps are skipped." -ForegroundColor Red
    "FAIL $msg" | Out-File $Log -Append -Encoding utf8; exit 1
}
function Run($exe, [string[]]$argv) {
    # Native tools print warnings on stderr (pip, torchaudio deprecations). Under
    # PowerShell 5 with 'Stop' those become terminating errors, so judge success
    # by the exit code only.
    $ErrorActionPreference = 'Continue'
    & $exe @argv 2>&1 | ForEach-Object { $l = "$_"; $l | Out-File $Log -Append -Encoding utf8; $l } | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail "$exe $($argv -join ' ') exited with $LASTEXITCODE" }
}
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}
function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function Winget-Install($id, $label) {
    if (-not (Have 'winget')) { Fail "$label is missing and winget isn't available. Install $label manually, then re-run." }
    Write-Host "  installing $label (a Windows prompt may ask for permission)..."
    Run 'winget' @('install', '--id', $id, '-e', '--accept-package-agreements', '--accept-source-agreements', '--silent')
    Refresh-Path
}

$T = 6

# 1. Prerequisites --------------------------------------------------------
Step 1 $T 'Checking Python 3.12 and Node.js'
$py = $null
if (Have 'py') { try { & py -3.12 -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @('py', '-3.12') } } catch {} }
if (-not $py) {
    Winget-Install 'Python.Python.3.12' 'Python 3.12'
    if (Have 'py') { $py = @('py', '-3.12') }
    elseif (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe") { $py = @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe") }
    else { Fail 'Python 3.12 installed but not found on PATH. Open a new terminal and re-run.' }
}
Ok "python: $($py -join ' ')"
if (-not (Have 'npm')) { Winget-Install 'OpenJS.NodeJS.LTS' 'Node.js LTS' }
if (-not (Have 'npm')) { Fail 'Node.js installed but npm not found. Open a new terminal and re-run.' }
Ok "node: $(node --version)"

# 2. Virtual environment --------------------------------------------------
Step 2 $T 'Creating the private Python environment (.venv)'
$Venv = Join-Path $Root '.venv'
$VPy = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $VPy)) {
    $exe = $py[0]; $rest = @(); if ($py.Count -gt 1) { $rest = $py[1..($py.Count - 1)] }
    Run $exe ($rest + @('-m', 'venv', $Venv))
}
Run $VPy @('-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', '--quiet')
Ok '.venv ready'

# 3. PyTorch ---------------------------------------------------------------
Step 3 $T 'Installing PyTorch'
$gpu = $null
# pip's 'scripts not on PATH' check can crash on untrusted PATH mount points (WinError 448).
$env:PIP_NO_WARN_SCRIPT_LOCATION = '1'
# Launched from the setup .exe, junctions on PATH (e.g. tools installed under
# %LOCALAPPDATA%\Programs pointing into dot-folders) raise WinError 448 inside pip.
# Install only needs real directories, so drop reparse points for this process.
$env:PATH = (($env:PATH -split ';') | Where-Object {
    $_ -and (Test-Path -LiteralPath $_ -PathType Container) -and
    -not ((Get-Item -LiteralPath $_ -Force -ErrorAction SilentlyContinue).Attributes -band [IO.FileAttributes]::ReparsePoint)
}) -join ';'
# nvidia-smi may be off PATH (or hidden by 32-bit redirection when launched by a 32-bit installer).
$Smi = (Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue).Source
if (-not $Smi) { foreach ($c in @("$env:WINDIR\Sysnative\nvidia-smi.exe", "$env:WINDIR\System32\nvidia-smi.exe")) { if (Test-Path $c) { $Smi = $c; break } } }
if (-not $Cpu -and $Smi) {
    try { $gpu = (& $Smi --query-gpu=name,memory.total --format=csv,noheader 2>$null | Select-Object -First 1) } catch {}
}
$hasTorch = $false
try { & $VPy -c "import torch" 2>$null; $hasTorch = ($LASTEXITCODE -eq 0) } catch {}
if ($hasTorch) { Ok 'already installed' }
elseif ($gpu) {
    Ok "NVIDIA GPU found: $gpu -> CUDA build"
    Run $VPy @('-m', 'pip', 'install', 'torch==2.8.0', 'torchaudio==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
} elseif ($Cpu) {
    Warn '-Cpu: test install only. The voice engine needs an NVIDIA GPU, so ATLAS will not be able to speak.'
    Run $VPy @('-m', 'pip', 'install', 'torch==2.8.0', 'torchaudio==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cpu')
} else {
    Fail ("No NVIDIA GPU found. ATLAS's voice engine needs an NVIDIA GPU (8 GB+ VRAM).`n" +
          "If you have one, update the driver from https://www.nvidia.com/drivers and run the installer again.")
}

# 4. ATLAS packages --------------------------------------------------------
Step 4 $T 'Installing ATLAS packages (this is the long one)'
Run $VPy @('-m', 'pip', 'install', '-r', (Join-Path $Root 'requirements-public.txt'))
Run $VPy @('-m', 'pip', 'install', '--no-deps', 'realtimestt==0.3.104')
Run $VPy @('-c', 'import fastapi, faster_whisper, RealtimeTTS, RealtimeSTT, speechbrain; print(''imports ok'')')
Ok 'packages ready'

# 5. Desktop shell + speech models ------------------------------------------
Step 5 $T 'Installing the desktop app and downloading speech models'
Push-Location (Join-Path $Root 'desktop')
try { Run 'npm.cmd' @('install', '--no-audit', '--no-fund', '--loglevel=error') } finally { Pop-Location }
Ok 'desktop shell ready'
if ($SkipModels) { Warn 'model download skipped (-SkipModels); ATLAS fetches them on first launch' }
else {
    $ErrorActionPreference = 'Continue'   # model hubs log progress on stderr
    & $VPy (Join-Path $Root 'tools\prefetch_models.py') 2>&1 | ForEach-Object { $l = "$_"; $l | Out-File $Log -Append -Encoding utf8; $l } | Out-Host
    if ($LASTEXITCODE -ne 0) { Warn 'some models will download on first launch instead' } else { Ok 'speech models downloaded' }
}
$ErrorActionPreference = 'Stop'
foreach ($d in 'user', 'voices', 'personas', 'logs') { New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null }

# 6. Shortcuts --------------------------------------------------------------
Step 6 $T 'Creating shortcuts'
$Electron = Join-Path $Root 'desktop\node_modules\electron\dist\electron.exe'
if (-not (Test-Path $Electron)) { Fail 'electron.exe missing after npm install' }
if ($NoShortcut) { Ok 'skipped (-NoShortcut)' }
else {
    $ws = New-Object -ComObject WScript.Shell
    $icon = Join-Path $Root 'static\favicon.ico'
    foreach ($dir in @([Environment]::GetFolderPath('Desktop'), (Join-Path ([Environment]::GetFolderPath('Programs')) ''))) {
        # Never take over a shortcut that belongs to a different ATLAS folder
        # (10-03: a test install overwrote the owner's dev-build shortcut).
        $path = Join-Path $dir 'ATLAS.lnk'
        if (Test-Path $path) {
            $old = $ws.CreateShortcut($path)
            $ours = Join-Path $Root 'desktop'
            if ($old.WorkingDirectory -and ($old.WorkingDirectory.TrimEnd('\') -ine $ours.TrimEnd('\')) -and (Test-Path $old.TargetPath)) {
                $path = Join-Path $dir ('ATLAS (' + (Split-Path $Root -Leaf) + ').lnk')
                Write-Host "    an ATLAS shortcut for another folder exists; leaving it alone, creating $(Split-Path $path -Leaf)"
            }
        }
        $lnk = $ws.CreateShortcut($path)
        $lnk.TargetPath = $Electron
        $lnk.Arguments = '.'
        $lnk.WorkingDirectory = Join-Path $Root 'desktop'
        if (Test-Path $icon) { $lnk.IconLocation = $icon }
        $lnk.Description = 'ATLAS voice agents'
        $lnk.Save()
    }
    Ok 'Desktop and Start menu shortcuts: ATLAS'
}

Write-Host ""
Write-Host "ATLAS is installed." -ForegroundColor Green
Write-Host "Open it from the ATLAS shortcut. First launch walks you through: language model (local or cloud),"
Write-Host "audio devices, importing a voice, and creating your first agent."
"=== done $(Get-Date -Format s) ===" | Out-File $Log -Append -Encoding utf8
