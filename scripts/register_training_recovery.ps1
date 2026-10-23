#Requires -Version 5.1
[CmdletBinding()]
param([string]$ProjectRoot = "", [switch]$StartNow)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$windowsPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
foreach ($path in @($python, (Join-Path $ProjectRoot 'scripts\start_curriculum.ps1'),
        (Join-Path $ProjectRoot 'scripts\start_training_monitor.ps1'), (Join-Path $ProjectRoot 'weiqi\training_watchdog.py'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file is missing: $path" }
}
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$service = New-Object -ComObject 'Schedule.Service'
$service.Connect()
$folder = $service.GetFolder('\')
$backup = Join-Path $ProjectRoot ('training_runs\curriculum\recovery-setup\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$null = New-Item -ItemType Directory -Path $backup -Force
$specs = @(
    @{Name='GoGoGo-KataGo-Training-Session'; Script='scripts\start_curriculum.ps1'},
    @{Name='GoGoGo-KataGo-Monitor-Session'; Script='scripts\start_training_monitor.ps1'},
    @{Name='GoGoGo-KataGo-Watchdog'; Script=$null}
)
foreach ($spec in $specs) {
    $oldTask = $null
    try { $oldTask = $folder.GetTask($spec.Name) } catch {}
    if ($oldTask) {
        [IO.File]::WriteAllText((Join-Path $backup ($spec.Name + '.xml')), $oldTask.Xml, [Text.Encoding]::Unicode)
        # Updating a task definition leaves the current running instance intact.
        $definition = $oldTask.Definition
    } else { $definition = $service.NewTask(0) }
    $definition.RegistrationInfo.Description = 'GoGoGo independent recovery: one-minute watchdog; checkpoint resume; explicit pauses respected.'
    $definition.Principal.UserId = $user
    $definition.Principal.LogonType = 3
    $definition.Principal.RunLevel = 0
    $definition.Actions.Clear()
    $action = $definition.Actions.Create(0)
    $action.WorkingDirectory = $ProjectRoot
    if ($spec.Script) {
        $action.Path = $windowsPowerShell
        $action.Arguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}" -ProjectRoot "{1}"' -f (Join-Path $ProjectRoot $spec.Script), $ProjectRoot
    } else {
        $action.Path = $windowsPowerShell
        $action.Arguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -Command "& ''{0}'' -m weiqi.training_watchdog check --project-root ''{1}''; exit $LASTEXITCODE"' -f $python.Replace("'", "''"), $ProjectRoot.Replace("'", "''")
        $definition.Triggers.Clear()
        $trigger = $definition.Triggers.Create(1)
        $trigger.StartBoundary = (Get-Date).AddMinutes(1).ToString("yyyy-MM-dd'T'HH:mm:sszzz", [Globalization.CultureInfo]::InvariantCulture)
        $trigger.Repetition.Interval = 'PT1M'
        $trigger.Repetition.StopAtDurationEnd = $false
        $trigger.Enabled = $true
        $logon = $definition.Triggers.Create(9)
        $logon.UserId = $user
        $logon.Enabled = $true
    }
    $settings = $definition.Settings
    $settings.Enabled = if ($oldTask -and $spec.Script) { $oldTask.Enabled } else { $true }
    $settings.AllowDemandStart = $true
    $settings.StartWhenAvailable = $true
    $settings.Hidden = $true
    $settings.MultipleInstances = 2
    $settings.ExecutionTimeLimit = 'PT0S'
    $settings.RestartInterval = 'PT1M'
    $settings.RestartCount = 999
    $settings.DisallowStartIfOnBatteries = $false
    $settings.StopIfGoingOnBatteries = $false
    $settings.RunOnlyIfNetworkAvailable = $false
    $settings.WakeToRun = $false
    $registered = $folder.RegisterTaskDefinition($spec.Name, $definition, 6, $user, $null, 3, $null)
    if ($StartNow -and $spec.Name -eq 'GoGoGo-KataGo-Watchdog') { $null = $registered.Run($null) }
    [pscustomobject]@{Task=$registered.Name;State=$registered.State;Enabled=$registered.Enabled;
        Triggers=$registered.Definition.Triggers.Count;Restart='1 minute';Backup=$backup}
}
