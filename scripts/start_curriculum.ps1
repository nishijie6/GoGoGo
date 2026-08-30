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

$launchMessage = "{0:o} starting curriculum (pid={1}, config={2})" -f `
    (Get-Date), $PID, $resolvedCurriculumConfig
Add-Content -LiteralPath $lifecycleLog -Value $launchMessage -Encoding UTF8

$exitCode = 1
Push-Location $resolvedProjectRoot
try {
    $env:PYTHONUNBUFFERED = "1"
    & $resolvedPythonPath -u $entryPoint curriculum `
        --curriculum-config $resolvedCurriculumConfig `
        1>> $stdoutLog 2>> $stderrLog

    if ($null -ne $LASTEXITCODE) {
        $exitCode = [int]$LASTEXITCODE
    }
}
catch {
    $errorMessage = "{0:o} launcher failure: {1}" -f (Get-Date), $_.Exception.Message
    Add-Content -LiteralPath $lifecycleLog -Value $errorMessage -Encoding UTF8
    $exitCode = 1
}
finally {
    Pop-Location
}

$exitMessage = "{0:o} curriculum exited with code {1}" -f (Get-Date), $exitCode
Add-Content -LiteralPath $lifecycleLog -Value $exitMessage -Encoding UTF8
exit $exitCode
