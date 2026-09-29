#Requires -Version 5.1

[CmdletBinding()]
param(
    [string]$TaskName = "GoGoGo-KataGo-Curriculum",
    [string]$ProjectRoot = "",
    [switch]$StartNow
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Task Scheduler 2.0 COM constants. COM is used instead of Register-ScheduledTask
# so the task can be installed by the current non-elevated user on stock Windows.
$taskTriggerLogon = 9
$taskTriggerTime = 1
$taskActionExec = 0
$taskCreateOrUpdate = 6
$taskLogonInteractiveToken = 3
$taskRunLevelLeastPrivilege = 0
$taskInstancesIgnoreNew = 2

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $scriptDirectory
}
$resolvedProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$startScript = Join-Path $resolvedProjectRoot "scripts\start_curriculum.ps1"
$entryPoint = Join-Path $resolvedProjectRoot "train_rl.py"
$pythonPath = Join-Path $resolvedProjectRoot ".venv\Scripts\python.exe"
$curriculumConfig = Join-Path $resolvedProjectRoot "config\rl_curriculum.rtx5070ti.json"

foreach ($requiredFile in @($startScript, $entryPoint, $pythonPath, $curriculumConfig)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Cannot register the curriculum task because this file is missing: $requiredFile"
    }
}

$windowsPowerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
if (-not (Test-Path -LiteralPath $windowsPowerShell -PathType Leaf)) {
    throw "Windows PowerShell was not found at $windowsPowerShell"
}

$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskService = New-Object -ComObject "Schedule.Service"
$taskService.Connect()
$rootFolder = $taskService.GetFolder("\")
$taskDefinition = $taskService.NewTask(0)

$taskDefinition.RegistrationInfo.Author = $currentUser
$taskDefinition.RegistrationInfo.Description = `
    "Resume the GoGoGo KataGo 9x9 -> 13x13 -> 19x19 curriculum after user logon."

$principal = $taskDefinition.Principal
$principal.Id = "CurrentUser"
$principal.UserId = $currentUser
$principal.LogonType = $taskLogonInteractiveToken
$principal.RunLevel = $taskRunLevelLeastPrivilege

$trigger = $taskDefinition.Triggers.Create($taskTriggerLogon)
$trigger.Id = "AtUserLogon"
$trigger.UserId = $currentUser
$trigger.Enabled = $true

# RestartOnFailure may not reliably restart a task that Windows reports as
# externally interrupted (0xC000013A). An indefinite five-minute time trigger
# acts as a watchdog for that case. MultipleInstances=IgnoreNew makes each
# trigger a no-op while the curriculum is already healthy and running.
$watchdogTrigger = $taskDefinition.Triggers.Create($taskTriggerTime)
$watchdogTrigger.Id = "FiveMinuteWatchdog"
$watchdogTrigger.StartBoundary = (Get-Date).AddMinutes(1).ToString(
    "yyyy-MM-dd'T'HH:mm:sszzz",
    [System.Globalization.CultureInfo]::InvariantCulture
)
$watchdogTrigger.Repetition.Interval = "PT5M"
$watchdogTrigger.Repetition.StopAtDurationEnd = $false
$watchdogTrigger.Enabled = $true

$action = $taskDefinition.Actions.Create($taskActionExec)
$action.Id = "RunCurriculum"
$action.Path = $windowsPowerShell
$action.Arguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $startScript
$action.WorkingDirectory = $resolvedProjectRoot

$settings = $taskDefinition.Settings
$settings.Enabled = $true
$settings.AllowDemandStart = $true
$settings.StartWhenAvailable = $true
$settings.Hidden = $true
$settings.MultipleInstances = $taskInstancesIgnoreNew
$settings.ExecutionTimeLimit = "PT0S"
$settings.RestartInterval = "PT5M"
$settings.RestartCount = 999
$settings.DisallowStartIfOnBatteries = $false
$settings.StopIfGoingOnBatteries = $false
$settings.RunOnlyIfNetworkAvailable = $false
$settings.WakeToRun = $false

$registeredTask = $rootFolder.RegisterTaskDefinition(
    $TaskName,
    $taskDefinition,
    $taskCreateOrUpdate,
    $currentUser,
    $null,
    $taskLogonInteractiveToken,
    $null
)

if ($StartNow) {
    $null = $registeredTask.Run($null)
}

$registeredDefinition = $registeredTask.Definition
[pscustomobject]@{
    TaskName = $registeredTask.Name
    User = $registeredDefinition.Principal.UserId
    Enabled = $registeredTask.Enabled
    Hidden = $registeredDefinition.Settings.Hidden
    Triggers = "At logon for $currentUser; five-minute watchdog"
    MultipleInstances = "IgnoreNew"
    RestartOnFailure = "Every 5 minutes, up to 999 times"
    ExecutionTimeLimit = $registeredDefinition.Settings.ExecutionTimeLimit
    Action = $registeredDefinition.Actions.Item(1).Path
    Arguments = $registeredDefinition.Actions.Item(1).Arguments
    StartedNow = [bool]$StartNow
} | Format-List
