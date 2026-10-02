#Requires -Version 5.1

[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$PythonPath = "",
    [string]$CurriculumConfig = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $scriptDirectory
}
$resolvedProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $resolvedProjectRoot ".venv\Scripts\python.exe"
}
$resolvedPythonPath = [System.IO.Path]::GetFullPath($PythonPath)

if ([string]::IsNullOrWhiteSpace($CurriculumConfig)) {
    $CurriculumConfig = Join-Path $resolvedProjectRoot "config\rl_curriculum.rtx5070ti.json"
}
$resolvedCurriculumConfig = [System.IO.Path]::GetFullPath($CurriculumConfig)
$entryPoint = Join-Path $resolvedProjectRoot "train_rl.py"
$logDirectory = Join-Path $resolvedProjectRoot "training_runs\curriculum\logs"

foreach ($requiredFile in @($resolvedPythonPath, $resolvedCurriculumConfig, $entryPoint)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required curriculum file was not found: $requiredFile"
    }
}

$null = New-Item -ItemType Directory -Path $logDirectory -Force
$launchTimestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutLog = Join-Path $logDirectory "curriculum-$launchTimestamp.stdout.log"
$stderrLog = Join-Path $logDirectory "curriculum-$launchTimestamp.stderr.log"
$lifecycleLog = Join-Path $logDirectory "launcher.log"

# The lifecycle log is advisory: a failed write (e.g. the file is briefly open
# elsewhere) is retried, then skipped, and never stops or fails the run.
function Write-LauncherLog([string]$Message) {
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            Add-Content -LiteralPath $lifecycleLog -Value $Message -Encoding UTF8 -ErrorAction Stop
            return
        }
        catch {
            Start-Sleep -Milliseconds 200
        }
    }
}

function Format-Argument([string]$Value) {
    return '"' + $Value.Replace('"', '\"') + '"'
}

Write-LauncherLog ("{0:o} starting curriculum (pid={1}, config={2})" -f `
    (Get-Date), $PID, $resolvedCurriculumConfig)

# Let the OS write the child's stdout and stderr straight to the log files.
# With PowerShell redirection (`1>> 2>>`) under Windows PowerShell 5.1, any
# line on stderr -- even a recoverable notice -- becomes a terminating error
# here, and a transient lock on the log file aborts the launcher; both used to
# end the training run. PYTHONUTF8 keeps the logs UTF-8 under every host.
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$arguments = @(
    "-u",
    (Format-Argument $entryPoint),
    "curriculum",
    "--curriculum-config",
    (Format-Argument $resolvedCurriculumConfig)
) -join " "

$exitCode = 1
try {
    $process = Start-Process -FilePath $resolvedPythonPath `
        -ArgumentList $arguments `
        -WorkingDirectory $resolvedProjectRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -NoNewWindow -PassThru
    # Reading Handle before waiting makes ExitCode available afterwards.
    $null = $process.Handle
    Write-LauncherLog ("{0:o} curriculum process started (pid={1})" -f (Get-Date), $process.Id)
    $process.WaitForExit()
    $exitCode = [int]$process.ExitCode
}
catch {
    Write-LauncherLog ("{0:o} launcher failure: {1}" -f (Get-Date), $_.Exception.Message)
    $exitCode = 1
}

Write-LauncherLog ("{0:o} curriculum exited with code {1}" -f (Get-Date), $exitCode)
exit $exitCode
