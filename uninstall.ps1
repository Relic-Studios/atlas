# ATLAS uninstaller (Windows). Easiest: double-click "Uninstall ATLAS" in the ATLAS folder or the Start menu,
# or use Windows Settings > Apps > ATLAS.
#
# Installed with the setup program (ATLAS-Setup-x.y.z.exe): this hands over to its uninstaller (unins000.exe),
# which removes the app and its Windows entry and calls back here (-FromInstaller) for what install.ps1 made.
# Installed from source (Install ATLAS.cmd): this removes everything itself.
#
# What goes: running ATLAS processes from this folder, the shortcuts that point here, the Windows "Apps" entry,
# the private Python environment (.venv), the desktop shell's packages, logs, and the app files. Your data
# (settings, voices, agents, memories, call recordings) is only deleted if you say so. Python, Node.js and the
# downloaded speech models (shared Hugging Face cache) are left alone; other programs may use them.
param([switch]$KeepData, [switch]$RemoveData, [switch]$FromInstaller, [switch]$Yes)

$ErrorActionPreference = 'Continue'
$Root = (Split-Path -Parent $MyInvocation.MyCommand.Path).TrimEnd('\')
$Unins = Join-Path $Root 'unins000.exe'
$DataDirs = @('user', 'private', 'voices', 'personas', 'agent_state', 'memory_db', 'recordings')
$AppsKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\ATLAS'

function Under($path) {
    if (-not $path) { return $false }
    try { $full = [IO.Path]::GetFullPath($path) } catch { return $false }
    return $full.StartsWith($Root + '\', [StringComparison]::OrdinalIgnoreCase) -or
           $full.Equals($Root, [StringComparison]::OrdinalIgnoreCase)
}

# Never the development folder (AGENTS.md and dev_pack are left out of every public build and install).
if ((Test-Path (Join-Path $Root 'AGENTS.md')) -or (Test-Path (Join-Path $Root 'dev_pack'))) {
    Write-Host "Refusing: $Root is the ATLAS development folder, not an install." -ForegroundColor Red
    exit 1
}

# Installed by the setup program: its uninstaller does the work (and calls back with -FromInstaller).
if (-not $FromInstaller -and (Test-Path $Unins)) {
    Start-Process -FilePath $Unins
    exit 0
}

if (-not $FromInstaller -and -not $Yes) {
    Write-Host "This removes ATLAS from $Root." -ForegroundColor Cyan
    if ((Read-Host 'Uninstall ATLAS? (y/n)') -notmatch '^[yY]') { Write-Host 'Nothing was changed.'; exit 1 }
}
if (-not $KeepData -and -not $RemoveData) {
    $a = Read-Host 'Also delete your ATLAS data (settings, voices, agents, memories, call recordings)? (y/n)'
    $RemoveData = $a -match '^[yY]'
}

# 1. Stop ATLAS if it's running from this folder (the desktop shell, the server, a memory pass).
$me = $PID
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessId -ne $me -and $_.Name -notin @('cmd.exe', 'powershell.exe', 'pwsh.exe') -and (
        (Under $_.ExecutablePath) -or
        ($_.Name -like 'python*.exe' -and $_.CommandLine -and
         ($_.CommandLine.IndexOf($Root, [StringComparison]::OrdinalIgnoreCase) -ge 0)))
} | ForEach-Object {
    Write-Host "  stopping $($_.Name) ($($_.ProcessId))"
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 800

# 2. Shortcuts that point into this folder (never another ATLAS install's).
$ws = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
    if (-not (Test-Path $dir)) { continue }
    Get-ChildItem -Path $dir -Filter '*.lnk' -ErrorAction SilentlyContinue | ForEach-Object {
        $l = $ws.CreateShortcut($_.FullName)
        if ((Under $l.TargetPath) -or (Under $l.WorkingDirectory)) {
            Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
            Write-Host "  removed shortcut $($_.Name)"
        }
    }
}

# 3. The Windows "Apps" entry install.ps1 wrote (the setup program removes its own).
if (Test-Path $AppsKey) {
    $loc = (Get-ItemProperty -Path $AppsKey -ErrorAction SilentlyContinue).InstallLocation
    if (Under $loc) { Remove-Item -Path $AppsKey -Recurse -Force -ErrorAction SilentlyContinue; Write-Host '  removed the Windows Apps entry' }
}

# 4. What install.ps1 and running ATLAS created.
foreach ($rel in @('.venv', 'desktop\node_modules', 'logs')) {
    $p = Join-Path $Root $rel
    if (Test-Path $p) { Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue; Write-Host "  removed $rel" }
}
Get-ChildItem -Path $Root -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Root -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'install*.log' -or $_.Name -in @('server.log', 'server.err.log', 'boot.pid') } |
    Remove-Item -Force -ErrorAction SilentlyContinue

# 5. Your data, only if you said so.
if ($RemoveData) {
    foreach ($d in $DataDirs) {
        $p = Join-Path $Root $d
        if (Test-Path $p) { Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue; Write-Host "  deleted your $d" }
    }
}

if ($FromInstaller) { exit 0 }      # the setup program's uninstaller removes the app files next

# 6. Installed from source: the app files. With your data kept, the folder stays with just your data in it.
if ($RemoveData) {
    # This script runs from the folder, so the folder itself goes a few seconds after this window closes.
    Start-Process -FilePath "$env:WINDIR\System32\cmd.exe" -WorkingDirectory $env:TEMP -WindowStyle Hidden `
        -ArgumentList '/c', "ping -n 6 127.0.0.1 >nul & rmdir /s /q `"$Root`""
    Write-Host ''
    Write-Host 'ATLAS is uninstalled. The folder is removed when this window closes.' -ForegroundColor Green
} else {
    Get-ChildItem -Path $Root -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notin $DataDirs -and $_.Name -ne 'uninstall.ps1' -and $_.Name -ne 'Uninstall ATLAS.cmd' } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Start-Process -FilePath "$env:WINDIR\System32\cmd.exe" -WorkingDirectory $env:TEMP -WindowStyle Hidden `
        -ArgumentList '/c', "ping -n 6 127.0.0.1 >nul & del /f /q `"$Root\uninstall.ps1`" `"$Root\Uninstall ATLAS.cmd`""
    Write-Host ''
    Write-Host "ATLAS is uninstalled. Your data is kept in $Root (delete that folder to remove it)." -ForegroundColor Green
}
exit 0
