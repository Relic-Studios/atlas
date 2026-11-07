# ATLAS's running processes for one install folder: the one rule the uninstaller and the setup program share.
# A process counts when it is not a command shell or this script and its program is under the folder, or when it is
# a python*.exe whose command line contains the folder (installs before 0.7 run the system Python from .venv).
#
#   powershell -File installer\atlas_processes.ps1 -Folder <folder>         one "<pid> <name>" line per process
#   powershell -File installer\atlas_processes.ps1 -Folder <folder> -Stop [-WaitSec 10]
#                         stops them, waits for them to exit, and prints the ones still running
#   . <path>\atlas_processes.ps1   then   Get-AtlasProcesses <folder> / Stop-AtlasProcesses <folder> [<seconds>]
# The exit code is always 0; the output is the answer. Dot-sourcing only defines the functions.
param([string]$Folder = '', [switch]$Stop, [int]$WaitSec = 10)

function Test-AtlasUnder([string]$Path, [string]$Root) {
    if (-not $Path -or -not $Root) { return $false }
    try { $full = [IO.Path]::GetFullPath($Path) } catch { return $false }
    return $full.StartsWith($Root + '\', [StringComparison]::OrdinalIgnoreCase) -or
           $full.Equals($Root, [StringComparison]::OrdinalIgnoreCase)
}

function Get-AtlasProcesses([string]$Root) {
    $Root = "$Root".TrimEnd('\')
    if (-not $Root) { return @() }
    $me = $PID
    return @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessId -ne $me -and $_.Name -notin @('cmd.exe', 'powershell.exe', 'pwsh.exe', '_unins.tmp') -and
        $_.Name -notlike 'unins*.exe' -and (
            (Test-AtlasUnder $_.ExecutablePath $Root) -or
            ($_.Name -like 'python*.exe' -and $_.CommandLine -and
             ($_.CommandLine.IndexOf($Root, [StringComparison]::OrdinalIgnoreCase) -ge 0)))
    })
}

# Stops them and waits up to $WaitSec for them to exit; returns the ones still running.
function Stop-AtlasProcesses([string]$Root, [int]$WaitSec = 5) {
    foreach ($p in (Get-AtlasProcesses $Root)) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    $deadline = (Get-Date).AddSeconds($WaitSec)
    do {
        $left = Get-AtlasProcesses $Root
        if (-not $left) { return @() }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    return $left
}

# Script mode (not when dot-sourced).
if ($MyInvocation.InvocationName -ne '.' -and $Folder) {
    $found = if ($Stop) { Stop-AtlasProcesses $Folder $WaitSec } else { Get-AtlasProcesses $Folder }
    foreach ($p in $found) { [Console]::Out.WriteLine(('{0} {1}' -f $p.ProcessId, $p.Name)) }
    exit 0
}
