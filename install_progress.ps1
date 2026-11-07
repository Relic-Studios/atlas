# Progress reporting for install.ps1 -Progress, the lines the ATLAS setup program (installer/atlas.iss) reads.
# Dot-sourced by install.ps1; loaded directly by tests/test_install_progress.py.
#
# Prints ASCII-only lines (Windows PowerShell 5.1 writes redirected output in the console's legacy code page):
#   @@ PID <n>                         first: this PowerShell's process id, for Stop
#   @@ LOG <file name>                 the log in the ATLAS folder this run appends to
#   @@ STEP <i>                        step i (1..5) starts
#   @@ P <overall>|<detail>|<minutes>  overall 0..1000 for the whole run, never decreasing; minutes left or empty
#   @@ WARN <code>                     MODELS: some models did not download
#   @@ FAIL <code>|<message>           before a non-zero exit (codes: NODISK LOCKED TLS SERVER NET FAILED)
#   @@ DONE                            last line of a successful run

# Steps 1 Python, 2 PyTorch, 3 packages, 4 models, 5 finishing. Weights: each step's share of the time at 100 Mbit/s,
# from a fresh install on 2026-11-06 (downloads scaled to 12.5 MB/s; installs as measured): 15 s, 552 s (232 s download,
# 320 s install), 399 s (27 s download, 361 s install, 11 s import check), 312 s, 1 s.
$script:AtWeights = @(1, 43, 31, 24, 1)
$script:AtTorchBytes = 2.9GB                         # step 2's download
$script:AtTorchInstalled = 4700000000                # what installing it writes (bytes, measured)
$script:AtPackageBytes = 330MB                       # step 3's downloads with progress lines, measured with an empty cache
$script:AtPackagesInstalled = 1000000000             # what installing them writes (bytes, measured)
$script:AtEnabled = $false
$script:AtStep = 0
$script:AtFraction = 0.0
$script:AtShown = -1
$script:AtDetail = ''
$script:AtLastDetail = $null
$script:AtMinutes = ''
$script:AtLastMinutes = $null
$script:AtLastPrint = [datetime]::MinValue
$script:AtStart = $null
$script:AtSamples = New-Object System.Collections.Generic.List[object]
$script:AtLastEstimate = [datetime]::MinValue
$script:AtPip = @{ Done = [double]0; Current = [double]0; File = ''; LastParsed = [datetime]::MinValue }
$script:AtTail = New-Object System.Collections.Generic.List[string]
$script:AtNow = { [datetime]::UtcNow }               # tests replace this to control the clock

function Enable-AtProgress { $script:AtEnabled = $true; $script:AtStart = & $script:AtNow }

function Write-AtLine([string]$s) {
    if (-not $script:AtEnabled) { return }
    $clean = -join ($s.ToCharArray() | ForEach-Object { if ([int]$_ -ge 32 -and [int]$_ -lt 127) { $_ } else { '?' } })
    [Console]::Out.WriteLine($clean)
    [Console]::Out.Flush()
}

function Get-AtOverall {
    $total = ($script:AtWeights | Measure-Object -Sum).Sum
    $done = 0
    for ($i = 1; $i -lt $script:AtStep; $i++) { $done += $script:AtWeights[$i - 1] }
    $cur = if ($script:AtStep -ge 1 -and $script:AtStep -le $script:AtWeights.Count) { $script:AtWeights[$script:AtStep - 1] * $script:AtFraction } else { 0 }
    return [int][math]::Floor(1000 * ($done + $cur) / $total)
}

# Minutes left from the rate of the last 60 s: first after 60 s, at most every 5 s.
function Update-AtMinutes([int]$Overall, [datetime]$Now) {
    $script:AtSamples.Add(@($Now, $Overall))
    while ($script:AtSamples.Count -gt 1 -and ($Now - $script:AtSamples[0][0]).TotalSeconds -gt 60) { $script:AtSamples.RemoveAt(0) }
    if (-not $script:AtStart -or ($Now - $script:AtStart).TotalSeconds -lt 60) { return }
    if (($Now - $script:AtLastEstimate).TotalSeconds -lt 5) { return }
    $first = $script:AtSamples[0]
    $secs = ($Now - $first[0]).TotalSeconds
    $gain = $Overall - $first[1]
    if ($secs -lt 10 -or $gain -le 0) { return }        # no rate yet: keep the last value
    $script:AtLastEstimate = $Now
    $est = [int][math]::Max(1, [math]::Ceiling((1000 - $Overall) / ($gain / $secs) / 60))
    $script:AtMinutes = [string][int]$est
}

function Publish-AtProgress([switch]$Force) {
    $o = Get-AtOverall
    if ($o -lt $script:AtShown) { $o = $script:AtShown }                  # never backwards
    $now = & $script:AtNow
    Update-AtMinutes $o $now
    if (-not $Force -and ($now - $script:AtLastPrint).TotalMilliseconds -lt 500) { return }
    if ($o -eq $script:AtShown -and $script:AtDetail -eq $script:AtLastDetail -and $script:AtMinutes -eq $script:AtLastMinutes) { return }
    $script:AtShown = $o
    $script:AtLastDetail = $script:AtDetail
    $script:AtLastMinutes = $script:AtMinutes
    $script:AtLastPrint = $now
    Write-AtLine ("@@ P {0}|{1}|{2}" -f $o, $script:AtDetail, $script:AtMinutes)
}

function Start-AtStep([int]$i) {
    $script:AtStep = $i
    $script:AtFraction = 0.0
    $script:AtDetail = ''
    $script:AtPip = @{ Done = [double]0; Current = [double]0; File = ''; LastParsed = (& $script:AtNow) }
    $script:AtTail.Clear()
    Write-AtLine "@@ STEP $i"
    Publish-AtProgress -Force
}

function Set-AtFraction([double]$f, $detail = $null) {   # untyped: a [string] parameter would turn $null into ''
    if ($null -ne $detail) { $detail = [string]$detail }
    $f = [math]::Max(0.0, [math]::Min(1.0, $f))
    if ($f -gt $script:AtFraction) { $script:AtFraction = $f }
    if ($null -ne $detail) { $script:AtDetail = $detail }
    Publish-AtProgress
}

function Complete-AtStep { $script:AtFraction = 1.0; Publish-AtProgress -Force }

function Format-AtBytes([double]$b) {
    if ($b -ge 1GB) { return ('{0:0.0} GB' -f ($b / 1GB)).Replace(',', '.') }
    return ('{0:0} MB' -f ($b / 1MB))
}

function Add-AtTail([string]$line) {
    $script:AtTail.Add($line)
    while ($script:AtTail.Count -gt 40) { $script:AtTail.RemoveAt(0) }
}

# One line of pip output (pip run with --progress-bar raw, through tools/pip_progress.py). ExpectedBytes: the step's
# download total. The downloads fill the first DownloadShare of the step; the install (INSTALLPROGRESS lines: bytes
# written while pip installs) fills the next InstallShare; whatever is left is for later work (step 3's import check).
function Read-AtPipLine([string]$line, [double]$ExpectedBytes, [string]$Label = '', [double]$DownloadShare = 1.0,
                        [double]$InstallShare = 0.0) {
    $p = $script:AtPip
    if ($line -match '^\s*Progress\s+(\d+)\s+of\s+(\d+)') {
        $p.Current = [double]$Matches[1]
        $p.LastParsed = & $script:AtNow
        if ($ExpectedBytes -gt 0) {
            $got = $p.Done + $p.Current
            $detail = if ($Label) { '{0} of {1}' -f (Format-AtBytes ([math]::Min($got, $ExpectedBytes))), (Format-AtBytes $ExpectedBytes) } else { $p.File }
            Set-AtFraction ($DownloadShare * [math]::Min(1.0, $got / $ExpectedBytes)) $detail
        }
        return
    }
    if ($line -match '^INSTALLPROGRESS\s+(\d+)\s+(\d+)') {
        $p.LastParsed = & $script:AtNow
        $exp = [double]$Matches[2]
        if ($exp -gt 0) {
            $detail = if ([double]$Matches[1] -ge $exp) { 'Finishing (can take several minutes)' } else { 'Installing' }
            Set-AtFraction ($DownloadShare + $InstallShare * [math]::Min(1.0, [double]$Matches[1] / $exp)) $detail
        }
        return
    }
    Add-AtTail $line
    if ($line -match '^\s*Downloading\s+(\S+)') {
        $p.Done += $p.Current; $p.Current = 0
        $p.File = ($Matches[1] -replace '-\d.*$', '')
        $p.LastParsed = & $script:AtNow
    } elseif ($line -match '^\s*(Installing collected packages|Building wheel|Successfully installed)') {
        $p.LastParsed = & $script:AtNow
        Set-AtFraction $DownloadShare 'Installing'           # the downloads are over, whatever the byte count said
    } elseif ($script:AtDetail -ne 'Installing' -and $script:AtDetail -notlike 'Finishing (can take*' -and $script:AtDetail -ne 'Downloading' -and
              (((& $script:AtNow) - $p.LastParsed).TotalSeconds -gt 60)) {
        Set-AtFraction $script:AtFraction 'Downloading'      # lines arrive but none parses: a changed pip format
    }
}

# A pip call has finished: its last file is complete (the next call's bytes add on top).
function Close-AtPipCall { $script:AtPip.Done += $script:AtPip.Current; $script:AtPip.Current = 0 }

# One line of tools/prefetch_models.py --progress output.
function Read-AtModelLine([string]$line) {
    if ($line -match '^MODELPROGRESS\s+(\d+)\s+(\d+)\s*(.*)$') {
        $done = [double]$Matches[1]; $exp = [double]$Matches[2]; $label = $Matches[3].Trim()
        if ($exp -gt 0) {
            $d = '{0} of {1}' -f (Format-AtBytes ([math]::Min($done, $exp))), (Format-AtBytes $exp)
            if ($label) { $d = "$label - $d" }
            Set-AtFraction ($done / $exp) $d
        }
        return
    }
    Add-AtTail $line
}

# Failure code from the last lines of output: the first match wins, in this order (TLS is case-sensitive).
function Get-AtFailureCode([string[]]$Lines) {
    $t = ($Lines -join "`n")
    if ($t -match '\[Errno 28\]|No space left on device|not enough space on the disk') { return 'NODISK' }
    if ($t -match '\[WinError 5\]|\[WinError 32\]|Access is denied|being used by another process|PermissionError') { return 'LOCKED' }
    if ($t -cmatch 'CERTIFICATE_VERIFY_FAILED|certificate verify failed|SSLCertVerificationError|WRONG_VERSION_NUMBER') { return 'TLS' }
    if ($t -match 'too many 5\d\d error responses|HTTP error 5\d\d|5\d\d Server Error|Service Unavailable|Bad Gateway|Gateway Timeout') { return 'SERVER' }
    if ($t -match 'Connection reset|Connection aborted|Connection refused|timed out|Max retries exceeded|NewConnectionError|getaddrinfo failed|Temporary failure in name resolution|ProxyError|ConnectionError|SSLEOFError|IncompleteRead') { return 'NET' }
    return 'FAILED'
}

function Get-AtTail { return @($script:AtTail) }

function Write-AtFail([string]$Code, [string]$Message) { Write-AtLine ("@@ FAIL {0}|{1}" -f $Code, $Message) }
