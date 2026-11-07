# ATLAS's running processes for one install folder: the one rule the uninstaller and the setup program share.
# A process counts when it is not a command shell or this script and its program is under the folder, or when it is
# a python*.exe with a rooted path argument inside it (older installs can run the system Python).
# Existing path components are expanded so short, long and mixed Windows spellings match.
#
#   powershell -File installer\atlas_processes.ps1 -Folder <folder>         one "<pid> <name>" line per process
#   powershell -File installer\atlas_processes.ps1 -Folder <folder> -Stop [-WaitSec 10]
#                         stops them, waits for them to exit, and prints the ones still running
#   . <path>\atlas_processes.ps1   then   Get-AtlasProcesses <folder> / Stop-AtlasProcesses <folder> [<seconds>]
# The exit code is always 0; the output is the answer. Dot-sourcing only defines the functions.
param([string]$Folder = '', [switch]$Stop, [int]$WaitSec = 10)

if (-not ('AtlasProcessPath' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
public static class AtlasProcessPath {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    private static extern uint GetLongPathName(string path, StringBuilder result, uint size);
    [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
    private static extern IntPtr CommandLineToArgvW(string commandLine, out int count);
    [DllImport("kernel32.dll")]
    private static extern IntPtr LocalFree(IntPtr memory);
    public static string Canonical(string path) {
        string full = Path.GetFullPath(path);
        var result = new StringBuilder(32768);
        uint size = GetLongPathName(full, result, (uint)result.Capacity);
        if (size == 0) {
            string parent = Path.GetDirectoryName(full);
            if (!String.IsNullOrEmpty(parent)) {
                size = GetLongPathName(parent, result, (uint)result.Capacity);
                if (size > 0 && size < result.Capacity)
                    return Path.Combine(result.ToString(), Path.GetFileName(full)).TrimEnd('\\');
            }
        }
        return (size > 0 && size < result.Capacity ? result.ToString() : full).TrimEnd('\\');
    }
    public static bool HasPathArgumentUnder(string commandLine, string root) {
        int count;
        IntPtr arguments = CommandLineToArgvW(commandLine, out count);
        if (arguments == IntPtr.Zero) return false;
        try {
            for (int i = 1; i < count; i++) {
                string value = Marshal.PtrToStringUni(Marshal.ReadIntPtr(arguments, i * IntPtr.Size));
                if (String.IsNullOrEmpty(value) || !Path.IsPathRooted(value)) continue;
                try {
                    string full = Canonical(value);
                    if (full.Equals(root, StringComparison.OrdinalIgnoreCase) ||
                        full.StartsWith(root + "\\", StringComparison.OrdinalIgnoreCase)) return true;
                } catch (ArgumentException) { }
                  catch (NotSupportedException) { }
                  catch (PathTooLongException) { }
            }
            return false;
        } finally { LocalFree(arguments); }
    }
}
'@
}

function Test-AtlasUnder([string]$Path, [string]$Root) {
    if (-not $Path -or -not $Root) { return $false }
    try { $full = [AtlasProcessPath]::Canonical($Path) } catch { return $false }
    return $full.StartsWith($Root + '\', [StringComparison]::OrdinalIgnoreCase) -or
           $full.Equals($Root, [StringComparison]::OrdinalIgnoreCase)
}

function Get-AtlasProcesses([string]$Root) {
    $Root = "$Root".TrimEnd('\')
    if (-not $Root) { return @() }
    try { $Root = [AtlasProcessPath]::Canonical($Root) } catch { return @() }
    $me = $PID
    return @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessId -ne $me -and $_.Name -notin @('cmd.exe', 'powershell.exe', 'pwsh.exe', '_unins.tmp') -and
        $_.Name -notlike 'unins*.exe' -and (
            (Test-AtlasUnder $_.ExecutablePath $Root) -or
            ($_.Name -like 'python*.exe' -and $_.CommandLine -and
             [AtlasProcessPath]::HasPathArgumentUnder($_.CommandLine, $Root)))
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
