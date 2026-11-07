# Setup-only GPU check for the ATLAS installer. Prints ONE verdict line and exits 0:
#   NOGPU | UNKNOWN <reason> | OLDDRIVER | OLDGPU <name> | LOWVRAM <name> <GB> | OK <name>
# Only positive evidence blocks Setup: whatever this script can't read gives UNKNOWN, and Setup continues.
# The checks run in this order and the first that applies is the verdict, so a card that is too old hears that
# (a driver update would not help) before an old driver is blamed.
# nvidia-smi runs hidden and is given up on, with its whole process tree, after -TimeoutSec, so a hung driver can't
# freeze Setup. -NvidiaSmi lets tests use a fake; normally it's found on PATH or in Sysnative / System32.
param([string]$NvidiaSmi = '', [int]$TimeoutSec = 10)

$ErrorActionPreference = 'Stop'
$MinCap = [decimal]'7.0'          # PyTorch 2.10's CUDA 12.8 build supports sm_70 to sm_120
$MinDriver = [version]'570.65'    # NVIDIA's Windows driver for CUDA 12.8 GA
$MinMiB = 7680                    # an "8 GB" card reports a little under 8192 MiB

function Say([string]$s) {
    $s = ($s -replace '[^\x20-\x7E]', '?') -replace '\s+', ' '
    if ($s.Length -gt 160) { $s = $s.Substring(0, 160) }
    [Console]::Out.WriteLine($s.Trim())
    exit 0
}

try {
    if (-not $NvidiaSmi) {
        $NvidiaSmi = (Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue).Source
        if (-not $NvidiaSmi) {
            foreach ($c in @("$env:WINDIR\Sysnative\nvidia-smi.exe", "$env:WINDIR\System32\nvidia-smi.exe")) {
                if (Test-Path -LiteralPath $c) { $NvidiaSmi = $c; break }
            }
        }
    }
    if (-not $NvidiaSmi -or -not (Test-Path -LiteralPath $NvidiaSmi)) { Say 'NOGPU' }

    $out = Join-Path ([IO.Path]::GetTempPath()) ("atlas_gpu_{0}.txt" -f [guid]::NewGuid().ToString('N'))
    $p = Start-Process -FilePath $NvidiaSmi -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError "$out.err" `
        -ArgumentList '--query-gpu=name,compute_cap,driver_version,memory.total', '--format=csv,noheader'
    $null = $p.Handle                 # keep the handle open, or ExitCode is empty after the wait
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        # the whole tree: a child left alive keeps the output pipe open and the caller would wait anyway. taskkill's
        # own complaint (the process ended just now) must not become an error here.
        $ErrorActionPreference = 'Continue'
        & "$env:WINDIR\System32\taskkill.exe" /T /F /PID $p.Id 2>&1 | Out-Null
        Say "UNKNOWN nvidia-smi did not answer within $TimeoutSec s"
    }
    $code = $p.ExitCode
    $text = ''
    foreach ($f in @($out, "$out.err")) {
        if (Test-Path -LiteralPath $f) { $text += [IO.File]::ReadAllText($f); Remove-Item -LiteralPath $f -ErrorAction SilentlyContinue }
    }
    if ($text -match 'No devices were found') { Say 'NOGPU' }
    if ($text -match 'not a valid field') { Say 'OLDDRIVER' }

    $inv = [Globalization.CultureInfo]::InvariantCulture
    $gpus = @()
    foreach ($line in ($text -split "`r?`n")) {
        $f = @($line -split ',' | ForEach-Object { $_.Trim() })
        if ($f.Count -lt 4 -or -not $f[0]) { continue }
        $cap = [decimal]0; $mib = 0
        if (-not [decimal]::TryParse($f[1], [Globalization.NumberStyles]::Number, $inv, [ref]$cap)) { continue }
        $drv = $null; try { $drv = [version]$f[2] } catch { continue }
        [void][int]::TryParse(($f[3] -replace '[^0-9]', ''), [ref]$mib)
        $gpus += [pscustomobject]@{ Name = $f[0]; Cap = $cap; Driver = $drv; MiB = $mib }
    }
    if (-not $gpus) {
        if ($code -ne 0) { Say "UNKNOWN nvidia-smi exited with code $code" }
        Say 'UNKNOWN no GPU in the nvidia-smi output'
    }
    $ok = @($gpus | Where-Object { $_.Cap -ge $MinCap } | Sort-Object -Property MiB -Descending)
    if (-not $ok) { Say ("OLDGPU " + $gpus[0].Name) }
    if ($ok[0].Driver -lt $MinDriver) { Say 'OLDDRIVER' }
    if ($ok[0].MiB -lt $MinMiB) { Say ("LOWVRAM {0} {1}" -f $ok[0].Name, [math]::Round($ok[0].MiB / 1024)) }
    Say ("OK " + $ok[0].Name)
} catch {
    Say ("UNKNOWN " + $_.Exception.Message)
}
