"""Independent Windows task watchdog; never infer trainer failure from GPU load."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

from .fileio import FileLockHeld, acquire_file_lock, atomic_write_json, release_file_lock
from .rl_handoff_guard import HandoffLockedError, ensure_training_available


TASKS = {"training": "GoGoGo-KataGo-Training-Session", "monitor": "GoGoGo-KataGo-Monitor-Session"}
PAUSE_FILE = "AUTO_RECOVERY_PAUSED"
POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"

SNAPSHOT_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$inputData = [Console]::In.ReadToEnd() | ConvertFrom-Json
$tasks = @{}
$service = New-Object -ComObject 'Schedule.Service'
$service.Connect()
foreach ($property in $inputData.tasks.PSObject.Properties) {
    try {
        $task = $service.GetFolder('\').GetTask([string]$property.Value)
        $epoch = if ($task.LastRunTime.Year -gt 2000) { ([DateTimeOffset]$task.LastRunTime).ToUnixTimeSeconds() } else { 0 }
        $tasks[$property.Name] = @{exists=$true; enabled=[bool]$task.Enabled; state=[int]$task.State; last_result=[long]$task.LastTaskResult; last_run=$epoch}
    } catch { $tasks[$property.Name] = @{exists=$false} }
}
$processes = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine } | ForEach-Object {
    @{pid=[int]$_.ProcessId; command=[string]$_.CommandLine}
})
@{tasks=$tasks; processes=$processes} | ConvertTo-Json -Depth 6 -Compress
'''

ACTION_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$inputData = [Console]::In.ReadToEnd() | ConvertFrom-Json
$service = New-Object -ComObject 'Schedule.Service'
$service.Connect()
$task = $service.GetFolder('\').GetTask([string]$inputData.name)
switch ($inputData.action) {
    'run' { $null = $task.Run($null) }
    'restart' { if ($task.State -eq 4) { $task.Stop(0) }; $null = $task.Run($null) }
    'pause' { $task.Enabled = $false; if ($task.State -eq 4) { $task.Stop(0) } }
    'resume' { $task.Enabled = $true; if ($task.State -ne 4) { $null = $task.Run($null) } }
    default { throw 'Unknown watchdog action' }
}
@{ok=$true; state=[int]$task.State} | ConvertTo-Json -Compress
'''


def powershell_json(script, payload):
    result = subprocess.run([POWERSHELL, "-NoLogo", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", script],
                            input=json.dumps(payload), text=True, encoding="utf-8", errors="replace",
                            capture_output=True, timeout=25, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"Windows task command exited {result.returncode}")
    return json.loads(result.stdout.lstrip("\ufeff"))


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except FileNotFoundError:
        return {}


def training_block(root, config_path=None):
    """Pause and handoff are authoritative, including markers in the active stage."""
    config_path = config_path or root / "config" / "rl_curriculum.rtx5070ti.json"
    config = read_json(config_path)
    if not config:
        return "课程配置不存在或为空"
    state_root = root / config.get("state_directory", "training_runs/curriculum")
    state = read_json(state_root / "state.json")
    directories = [state_root]
    board = state.get("active_board_size", 9)
    for stage in config.get("stages", []):
        if stage.get("board_size") == board:
            profile = read_json(root / stage["training_config"])
            run = profile.get("overrides", {}).get("runtime", {}).get("output_directory")
            if run:
                directories.append(root / run)
    for directory in directories:
        for marker in (PAUSE_FILE, "STOP", "PAUSE"):
            if (directory / marker).exists():
                return f"主动暂停标记：{directory / marker}"
        try:
            ensure_training_available(directory)
        except HandoffLockedError as error:
            return str(error)
    return None


def decide(task, pids, previous, now, *, blocked=None, web_ok=True):
    """Pure recovery policy, shared by production and failure-injection tests."""
    result = dict(previous)
    result.update(pids=pids, action=None, task_state=task.get("state"))
    if blocked:
        return {**result, "state": "paused", "message": blocked}
    if not task.get("exists"):
        return {**result, "state": "failed", "message": "Windows 后台任务不存在，需要重新注册"}
    if not task.get("enabled"):
        return {**result, "state": "paused", "message": "后台任务已手动禁用"}
    if pids:
        failures = 0 if web_ok else previous.get("health_failures", 0) + 1
        if failures < 3:
            return {**result, "state": "healthy" if web_ok else "checking", "health_failures": failures,
                    "consecutive_failures": 0, "message": "进程正常" if web_ok else "正在复核监控网页连接"}
    else:
        failures = 0
        if task.get("state") == 4 and now - task.get("last_run", 0) < 90:
            return {**result, "state": "starting", "message": "后台任务正在启动，等待进程就绪"}
        # A Task Scheduler Stop is an intentional stop; a crashed process has
        # its actual exit code instead. Never fight a user's manual pause.
        if task.get("last_result") == 0x41306 and previous.get("state") not in {"restarting", "failed", "retry_wait"}:
            return {**result, "state": "paused", "message": "后台任务被手动停止，等待显式恢复"}
    attempts = previous.get("consecutive_failures", 0)
    cooldown = min(900, 60 * 2 ** min(max(0, attempts - 1), 4))
    if now - previous.get("last_restart_epoch", 0) < cooldown:
        return {**result, "state": "retry_wait", "message": "等待下一次恢复重试", "health_failures": failures}
    return {**result, "state": "restarting", "message": "检测到进程退出，正在自动恢复" if not pids else "监控网页连续失联，正在单独恢复",
            "action": "restart" if task.get("state") == 4 else "run", "health_failures": failures,
            "consecutive_failures": attempts + 1, "last_restart_epoch": now,
            "restart_count": previous.get("restart_count", 0) + 1}


def matching_pids(root, processes, kind):
    script = str(root / ("train_rl.py" if kind == "training" else "monitor_training.py")).casefold()
    matches = []
    for item in processes:
        command = item.get("command", "").casefold()
        if script not in command:
            continue
        remainder = command.split(script, 1)[1].lstrip('"\' \t').split()
        if kind == "monitor" or (remainder and remainder[0] == "curriculum"):
            matches.append(item["pid"])
    return matches


def check(root, *, tasks=None, snapshot=None, action_runner=None, web_ok=None, now=None):
    tasks = tasks or TASKS
    now = time.time() if now is None else now
    state_root = root / "training_runs" / "curriculum"
    state_root.mkdir(parents=True, exist_ok=True)
    handle = acquire_file_lock(state_root / "recovery.lock")
    try:
        previous = read_json(state_root / "recovery_status.json")
        snapshot = snapshot or powershell_json(SNAPSHOT_SCRIPT, {"tasks": tasks})
        action_runner = action_runner or (lambda name, action: powershell_json(ACTION_SCRIPT, {"name": name, "action": action}))
        if web_ok is None:
            try:
                with urlopen("http://127.0.0.1:8766/api/health", timeout=2) as response:
                    web_ok = response.status == 200 and json.load(response).get("ok") is True
            except (OSError, ValueError):
                web_ok = False
        blocked = training_block(root)
        events = list(previous.get("events", []))[-39:]
        stamp = datetime.fromtimestamp(now, timezone.utc).astimezone().isoformat(timespec="seconds")
        status = {"schema_version": 1, "checked_at": stamp, "checked_epoch": now, "interval_seconds": 60, "events": events}
        for kind, name in tasks.items():
            old = previous.get(kind, {})
            target = decide(snapshot["tasks"].get(kind, {}), matching_pids(root, snapshot.get("processes", []), kind), old, now,
                            blocked=blocked if kind == "training" else None, web_ok=web_ok if kind == "monitor" else True)
            if target["state"] == "healthy" and old.get("state") in {"restarting", "starting", "failed", "retry_wait"} and old.get("restart_count", 0):
                target["message"] = "自动恢复完成，进程正常"
            action = target.pop("action", None)
            if action:
                # Save the recovery intention before invoking Windows. Even a
                # failed task start leaves a visible record and bounded backoff.
                status[kind] = target
                atomic_write_json(state_root / "recovery_status.json", {**previous, **status})
                try:
                    action_runner(name, action)
                except Exception as error:
                    target.update(state="failed", message=f"自动恢复失败：{error}")
            status[kind] = target
            changed = target.get("state") != old.get("state") or action is not None
            if changed:
                event = {"timestamp": stamp, "service": kind, "state": target["state"], "message": target["message"]}
                events.append(event)
                with (state_root / "recovery_events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        status["events"] = events[-40:]
        atomic_write_json(state_root / "recovery_status.json", status)
        return status
    finally:
        release_file_lock(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Windows 围棋训练自动恢复守护")
    parser.add_argument("command", choices=("check", "pause", "resume", "can-start"), nargs="?", default="check")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--curriculum-config", type=Path)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    pause = root / "training_runs" / "curriculum" / PAUSE_FILE
    try:
        if args.command == "can-start":
            reason = training_block(root, args.curriculum_config)
            if reason:
                print(reason)
                return 10
            return 0
        if args.command == "pause":
            pause.parent.mkdir(parents=True, exist_ok=True)
            pause.write_text("manual pause\n", encoding="utf-8")
            powershell_json(ACTION_SCRIPT, {"name": TASKS["training"], "action": "pause"})
        elif args.command == "resume":
            reason = training_block(root)
            if reason and not reason.startswith("主动暂停标记："):
                raise RuntimeError(reason)
            pause.unlink(missing_ok=True)
            reason = training_block(root)
            if reason:
                raise RuntimeError(reason)
            powershell_json(ACTION_SCRIPT, {"name": TASKS["training"], "action": "resume"})
        result = check(root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except FileLockHeld:
        return 0
    except Exception as error:
        print(f"恢复守护错误：{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
