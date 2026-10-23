#Requires -Version 5.1
[CmdletBinding()]
param([string]$ProjectRoot = "")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$entry = Join-Path $ProjectRoot 'monitor_training.py'
$logs = Join-Path $ProjectRoot 'training_runs\curriculum'
$null = New-Item -ItemType Directory -Path $logs -Force
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$arguments = '-u "{0}" --run curriculum' -f $entry
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput (Join-Path $logs "monitor-task-$stamp.out.log") `
    -RedirectStandardError (Join-Path $logs "monitor-task-$stamp.err.log") -WindowStyle Hidden -PassThru
$null = $process.Handle
$process.WaitForExit()
exit $process.ExitCode
