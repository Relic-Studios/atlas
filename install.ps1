# ATLAS installer (Windows 10/11). Run from the ATLAS folder:
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# Everything is local and account-free: no API keys, no sign-ins. Five steps:
#   1. Python: a private environment in .venv (never touches system Python)
#   2. PyTorch (CUDA build; an NVIDIA GPU is required for the voice engine)
#   3. ATLAS's Python packages
#   4. The speech and voice models, so the first launch downloads nothing
#   5. The desktop shell, shortcuts, and (source installs) the Windows Apps entry
# Re-running is safe: finished steps are skipped. Choosing a language model (local via Ollama, or a cloud provider
# with your own key) happens in the first-run setup inside the app.
#
# A run without -Runtime is a source install (a git checkout, "Install ATLAS.cmd"): Python 3.12 and Node.js come from
# winget if missing, npm installs the desktop shell, and the script checks the GPU itself. The setup program
# (ATLAS-Setup-x.y.z.exe) passes:
#   -Runtime <python.exe>  make .venv from Setup's bundled Python; no winget, Node.js or npm (the shell is prebuilt)
#   -Progress              print the "@@" lines Setup's wizard reads (install_progress.ps1)
#   -NoDesktopShortcut     the user unticked the desktop shortcut
#   -GpuChecked            Setup already checked the GPU: install the CUDA build without asking nvidia-smi
param([switch]$Cpu, [switch]$SkipModels, [switch]$NoShortcut,
      [string]$Runtime = '', [switch]$Progress, [switch]$NoDesktopShortcut, [switch]$GpuChecked)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Source = -not $Runtime
$PipPin = '26.3'          # pinned: Setup's progress parser reads pip's raw progress lines
$T = 5
. (Join-Path $Root 'install_progress.ps1')
if ($Progress) { Enable-AtProgress; Write-AtLine "@@ PID $PID" }

# Each run appends to the log after a dated line, so an earlier failed run stays readable.
$Log = Join-Path $Root 'install.log'
$Stamp = "=== ATLAS install $(Get-Date -Format s) ==="
try { $Stamp | Out-File $Log -Append -Encoding utf8 }
catch {  # log held open by an earlier, still-running install: use a fresh file
    $Log = Join-Path $Root ("install-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    $Stamp | Out-File $Log -Encoding utf8
}
if ($Progress) { Write-AtLine ("@@ LOG " + (Split-Path $Log -Leaf)) }

$script:Tail = New-Object System.Collections.Generic.List[string]
function Add-Tail($l) { $script:Tail.Add($l); while ($script:Tail.Count -gt 40) { $script:Tail.RemoveAt(0) } }
function Step($n, $msg) {
    Write-Host ""; Write-Host "[$n/$T] $msg" -ForegroundColor Cyan; "[$n/$T] $msg" | Out-File $Log -Append -Encoding utf8
    Start-AtStep $n
}
function Ok($msg)   { Write-Host "  ok  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  !!  $msg" -ForegroundColor Yellow; "WARN $msg" | Out-File $Log -Append -Encoding utf8 }
function Fail($msg) {
    Write-Host ""; Write-Host "  Install stopped: $msg" -ForegroundColor Red
    Write-Host "  Details are in $Log. Fix the problem above and run install.ps1 again; finished steps are skipped." -ForegroundColor Red
    "FAIL $msg" | Out-File $Log -Append -Encoding utf8
    Write-AtFail (Get-AtFailureCode @($script:Tail)) $msg
    exit 1
}
# Runs a native tool; every line goes to the log and the console, and to $OnLine (Setup's progress parser).
# pip's "Progress <n> of <m>" lines and pip_progress.py's INSTALLPROGRESS lines only go to $OnLine. Native tools print warnings on stderr (pip, torchaudio
# deprecations); under PowerShell 5 with 'Stop' those become terminating errors, so success is the exit code only.
function Run($exe, [string[]]$argv, [scriptblock]$OnLine = $null) {
    $ErrorActionPreference = 'Continue'
    $script:Tail.Clear()
    & $exe @argv 2>&1 | ForEach-Object {
        $l = "$_"
        if ($OnLine) { & $OnLine $l }
        if ($l -notmatch '^\s*Progress\s+\d+\s+of\s+\d+|^INSTALLPROGRESS ') {
            $l | Out-File $Log -Append -Encoding utf8
            Add-Tail $l
            $l
        }
    } | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail "$exe $($argv -join ' ') exited with $LASTEXITCODE" }
}
# Runs a native tool for its answer only (no log, no failure).
function Probe($exe, [string[]]$argv) {
    $ErrorActionPreference = 'Continue'
    try { $o = & $exe @argv 2>$null; if ($LASTEXITCODE -eq 0) { return "$o".Trim() } } catch {}
    return $null
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

$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
# pip's 'scripts not on PATH' check can crash on untrusted PATH mount points (WinError 448).
$env:PIP_NO_WARN_SCRIPT_LOCATION = '1'
# Launched from the setup .exe, junctions on PATH (e.g. tools installed under %LOCALAPPDATA%\Programs pointing into
# dot-folders) raise WinError 448 inside pip. Install only needs real directories, so drop reparse points here.
$env:PATH = (($env:PATH -split ';') | Where-Object {
    $_ -and (Test-Path -LiteralPath $_ -PathType Container) -and
    -not ((Get-Item -LiteralPath $_ -Force -ErrorAction SilentlyContinue).Attributes -band [IO.FileAttributes]::ReparsePoint)
}) -join ';'
$Pb = @(); if ($Progress) { $Pb = @('--progress-bar', 'raw') }
# Under -Progress, a pip install runs through tools/pip_progress.py, which adds INSTALLPROGRESS lines (bytes written while
# pip installs, which it does silently for minutes) so the setup program's bar keeps moving.
function PipArgs([string[]]$pipArgs, [long]$expectedInstall) {
    if ($Progress) {
        return @((Join-Path $Root 'tools\pip_progress.py'), '--path', $Venv, '--expected', "$expectedInstall", '--',
                 $VPy, '-m', 'pip') + $pipArgs + $Pb
    }
    return @('-m', 'pip') + $pipArgs
}

# Room for the Python environment, PyTorch and the models (Setup checks every drive itself).
if (-not $Progress) {
    try {
        $free = (Get-PSDrive -Name ($Root.Substring(0, 1))).Free / 1GB
        if ($free -lt 20) { Warn ("Only {0:N0} GB free on drive {1}: ATLAS needs about 15 GB, plus room for your recordings." -f $free, $Root.Substring(0, 1)) }
    } catch {}
}

# 1. Python ----------------------------------------------------------------
Step 1 "Preparing ATLAS's Python"
if ($Runtime) {
    if (-not (Test-Path -LiteralPath $Runtime)) { Fail "The bundled Python is missing: $Runtime" }
    $py = @($Runtime)
} else {
    $py = $null
    if ((Have 'py') -and ((Probe 'py' @('-3.12', '-c', 'print(1)')) -eq '1')) { $py = @('py', '-3.12') }
    if (-not $py) {
        Winget-Install 'Python.Python.3.12' 'Python 3.12'
        if (Have 'py') { $py = @('py', '-3.12') }
        elseif (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe") { $py = @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe") }
        else { Fail 'Python 3.12 installed but not found on PATH. Open a new terminal and re-run.' }
    }
    if (-not (Have 'npm')) { Winget-Install 'OpenJS.NodeJS.LTS' 'Node.js LTS' }
    if (-not (Have 'npm')) { Fail 'Node.js installed but npm not found. Open a new terminal and re-run.' }
    Ok "node: $(node --version)"
}
$pyExe = $py[0]; $pyRest = @(); if ($py.Count -gt 1) { $pyRest = $py[1..($py.Count - 1)] }
Ok "python: $($py -join ' ')"
$Venv = Join-Path $Root '.venv'
$VPy = Join-Path $Venv 'Scripts\python.exe'
# Keep a .venv that runs and was made from the same Python minor version (an older Setup made it from the system
# Python 3.12: still fine); otherwise make it again. pip's wheel cache makes a rebuild mostly local.
$baseVer = Probe $pyExe ($pyRest + @('-c', "import sys; print('%d.%d' % sys.version_info[:2])"))
$keep = $false
if (Test-Path $VPy) {
    $venvVer = ''
    $cfg = Join-Path $Venv 'pyvenv.cfg'
    if (Test-Path $cfg) {
        $m = Select-String -Path $cfg -Pattern '^\s*version(_info)?\s*=\s*(\d+\.\d+)' | Select-Object -First 1
        if ($m) { $venvVer = $m.Matches[0].Groups[2].Value }
    }
    $keep = ((Probe $VPy @('-c', 'print(1)')) -eq '1') -and $venvVer -and ($venvVer -eq $baseVer)
    if ($keep) { Ok ".venv kept (Python $venvVer)" } else { Warn ".venv doesn't run or is for another Python ($venvVer vs $baseVer); making it again" }
}
if (-not $keep) { Run $pyExe ($pyRest + @('-m', 'venv', '--clear', $Venv)) }
Set-AtFraction 0.5 ''
Run $VPy @('-m', 'pip', 'install', "pip==$PipPin", 'wheel', 'setuptools>=83', '--quiet')
Ok '.venv ready'
Complete-AtStep

# 2. PyTorch -----------------------------------------------------------------
Step 2 'Downloading PyTorch'
$gpu = $null
if (-not $GpuChecked -and -not $Cpu) {
    # nvidia-smi may be off PATH (or hidden by 32-bit redirection when launched by a 32-bit installer).
    $Smi = (Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue).Source
    if (-not $Smi) { foreach ($c in @("$env:WINDIR\Sysnative\nvidia-smi.exe", "$env:WINDIR\System32\nvidia-smi.exe")) { if (Test-Path $c) { $Smi = $c; break } } }
    if ($Smi) { try { $gpu = (& $Smi --query-gpu=name,memory.total --format=csv,noheader 2>$null | Select-Object -First 1) } catch {} }
}
# Pinned PyTorch (known advisories: SECURITY.md). Upgrades keep .venv, so an older torch there is replaced, not
# skipped; a CUDA build stays CUDA and a CPU test build stays CPU.
$TorchPin = '2.10.0'
$torchVer = "$(Probe $VPy @('-W', 'ignore', '-c', 'import torch; print(torch.__version__)'))".Trim()
$cudaTorch = ($torchVer -like '*+cu*') -and -not $Cpu
$torchLines = { param($l) Read-AtPipLine $l $script:AtTorchBytes 'PyTorch' 0.42 0.58 }   # 232 s download, 320 s install
if ($torchVer -eq $TorchPin -or $torchVer -like "$TorchPin+*") { Ok "already installed ($torchVer)" }
elseif (($GpuChecked -and -not $Cpu) -or $gpu -or $cudaTorch) {
    if ($torchVer) { Ok "updating PyTorch $torchVer -> $TorchPin" }
    if ($gpu) { Ok "NVIDIA GPU found: $gpu -> CUDA build" }
    Run $VPy (PipArgs @('install', "torch==$TorchPin", "torchaudio==$TorchPin", '--index-url', 'https://download.pytorch.org/whl/cu128') $script:AtTorchInstalled) $torchLines
} elseif ($Cpu -or $torchVer) {
    Warn 'CPU build: test install only. The voice engine needs an NVIDIA GPU, so ATLAS will not be able to speak.'
    Run $VPy (PipArgs @('install', "torch==$TorchPin", "torchaudio==$TorchPin", '--index-url', 'https://download.pytorch.org/whl/cpu') $script:AtTorchInstalled) $torchLines
} else {
    Fail ("No NVIDIA GPU found. ATLAS's voice engine needs an NVIDIA GPU (8 GB+ VRAM).`n" +
          "If you have one, update the driver from https://www.nvidia.com/drivers and run the installer again.")
}
Complete-AtStep

# 3. ATLAS packages ----------------------------------------------------------
Step 3 "Installing ATLAS's packages"
$pkgLines = { param($l) Read-AtPipLine $l $script:AtPackageBytes 'packages' 0.07 0.85 }   # 27 s, 361 s; the rest: import check
Run $VPy (PipArgs @('install', '-r', (Join-Path $Root 'requirements-public.txt')) $script:AtPackagesInstalled) $pkgLines
Close-AtPipCall
Run $VPy (@('-m', 'pip', 'install', '--no-deps', 'realtimestt==0.3.104') + $Pb) $pkgLines
Set-AtFraction 0.92 'Checking the packages'
Run $VPy @('-c', 'import fastapi, faster_whisper, RealtimeTTS, RealtimeSTT, speechbrain; print(''imports ok'')')
Ok 'packages ready'
Complete-AtStep

# 4. Speech and voice models ---------------------------------------------------
Step 4 'Downloading speech and voice models'
if ($SkipModels) {
    Warn 'model download skipped (-SkipModels); ATLAS fetches them on first launch'
    Write-AtLine '@@ WARN MODELS'
} else {
    $ErrorActionPreference = 'Continue'   # model hubs log progress on stderr
    $margs = @((Join-Path $Root 'tools\prefetch_models.py'))
    if ($Progress) { $margs += '--progress'; $env:HF_HUB_DISABLE_PROGRESS_BARS = '1' }
    & $VPy @margs 2>&1 | ForEach-Object {
        $l = "$_"
        if ($l -like 'MODELPROGRESS *') { Read-AtModelLine $l }
        else { $l | Out-File $Log -Append -Encoding utf8; $l }
    } | Out-Host
    if ($LASTEXITCODE -ne 0) { Warn 'some models will download on first launch instead'; Write-AtLine '@@ WARN MODELS' }
    else { Ok 'speech and voice models downloaded' }
    $ErrorActionPreference = 'Stop'
}
Complete-AtStep

# 5. Desktop shell, shortcuts, Apps entry ---------------------------------------
Step 5 'Finishing'
if ($Source) {
    Push-Location (Join-Path $Root 'desktop')
    try {
        Run 'npm.cmd' @('install', '--no-audit', '--no-fund', '--loglevel=error')
        # Electron 45 no longer downloads its binary during npm install; fetch it now
        # (a no-op when it's already there) so the shortcut below has electron.exe.
        Run 'node' @('node_modules\electron\install.js')
    } finally { Pop-Location }
    Ok 'desktop shell ready'
}
foreach ($d in 'user', 'voices', 'personas', 'logs') { New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null }
$Electron = Join-Path $Root 'desktop\node_modules\electron\dist\electron.exe'
if (-not (Test-Path $Electron)) { Fail "ATLAS's desktop app is missing (desktop\node_modules\electron\dist\electron.exe)" }
if ($NoShortcut) { Ok 'shortcuts skipped (-NoShortcut)' }
else {
    $ws = New-Object -ComObject WScript.Shell
    $icon = Join-Path $Root 'static\favicon.ico'
    $dirs = @((Join-Path ([Environment]::GetFolderPath('Programs')) ''))
    if (-not $NoDesktopShortcut) { $dirs = @([Environment]::GetFolderPath('Desktop')) + $dirs }
    foreach ($dir in $dirs) {
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
    if ($NoDesktopShortcut) { Ok 'Start menu shortcut: ATLAS' } else { Ok 'Desktop and Start menu shortcuts: ATLAS' }
    # The setup program makes its own "Uninstall ATLAS" (its styled uninstaller); a source install gets this one.
    $UninstCmd = Join-Path $Root 'Uninstall ATLAS.cmd'
    if ($Source -and (Test-Path $UninstCmd)) {
        $lnk = $ws.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Programs')) 'Uninstall ATLAS.lnk'))
        $lnk.TargetPath = $UninstCmd
        $lnk.WorkingDirectory = $Root
        if (Test-Path $icon) { $lnk.IconLocation = $icon }
        $lnk.Description = 'Uninstall ATLAS'
        $lnk.Save()
        Ok 'Start menu: Uninstall ATLAS'
    }
}

# Windows Settings > Apps entry, so ATLAS can be uninstalled like any program. The setup program
# (unins000.exe in this folder, its *_is1 key) registers its own entry; a source install gets this one.
$SetupEntry = Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall' -ErrorAction SilentlyContinue |
    Where-Object { $_.PSChildName -like '*_is1' -and
                   ([string](Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue).InstallLocation).TrimEnd('\') -ieq $Root }
if ($Source -and (Test-Path (Join-Path $Root 'uninstall.ps1')) -and -not (Test-Path (Join-Path $Root 'unins000.exe')) -and -not $SetupEntry) {
    $ver = '0'
    $cl = Join-Path $Root 'CHANGELOG.md'
    if (Test-Path $cl) { $m = Select-String -Path $cl -Pattern '^##\s+([0-9][0-9.]*)' | Select-Object -First 1; if ($m) { $ver = $m.Matches[0].Groups[1].Value } }
    $key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\ATLAS'
    try {
        New-Item -Path $key -Force | Out-Null
        $ps = "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe"
        New-ItemProperty -Path $key -Name 'DisplayName' -Value 'ATLAS' -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $key -Name 'DisplayVersion' -Value $ver -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $key -Name 'Publisher' -Value 'Relic Studios' -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $key -Name 'InstallLocation' -Value $Root -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $key -Name 'DisplayIcon' -Value (Join-Path $Root 'static\favicon.ico') -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $key -Name 'UninstallString' -PropertyType String -Force `
            -Value ('"' + $ps + '" -NoProfile -ExecutionPolicy Bypass -File "' + (Join-Path $Root 'uninstall.ps1') + '"') | Out-Null
        New-ItemProperty -Path $key -Name 'NoModify' -Value 1 -PropertyType DWord -Force | Out-Null
        New-ItemProperty -Path $key -Name 'NoRepair' -Value 1 -PropertyType DWord -Force | Out-Null
        Ok 'Windows Settings > Apps: ATLAS (uninstall from there or with "Uninstall ATLAS")'
    } catch { Warn "Couldn't add ATLAS to Windows Settings > Apps ($_). Uninstall with ""Uninstall ATLAS.cmd"" in $Root." }
}
Complete-AtStep

Write-Host ""
Write-Host "ATLAS is installed." -ForegroundColor Green
Write-Host "Open it from the ATLAS shortcut. First launch walks you through: language model (local or cloud),"
Write-Host "audio devices, and meeting Juniper, your ready-to-use agent and voice. Customization is optional."
Write-Host "To uninstall: Windows Settings > Apps > ATLAS, or ""Uninstall ATLAS"" in the Start menu or in $Root."
"=== done $(Get-Date -Format s) ===" | Out-File $Log -Append -Encoding utf8
Write-AtLine '@@ DONE'
