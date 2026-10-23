"""Recovery policy and persisted failures, isolated from the real trainer."""

import json
from pathlib import Path
import tempfile
import unittest

from weiqi.training_watchdog import check, decide, matching_pids, training_block


def ready(**overrides):
    return {"exists": True, "enabled": True, "state": 3, "last_result": 1, "last_run": 0, **overrides}


class WatchdogPolicyTests(unittest.TestCase):
    def test_running_training_is_never_restarted_for_old_logs_or_low_gpu_load(self):
        result = decide(ready(state=4), [123], {}, 1000)
        self.assertEqual(result["state"], "healthy")
        self.assertIsNone(result["action"])

    def test_crash_and_orphaned_launcher_recover_but_startup_has_grace(self):
        self.assertEqual(decide(ready(), [], {}, 1000)["action"], "run")
        self.assertEqual(decide(ready(state=4, last_run=950), [], {}, 1000)["state"], "starting")
        self.assertEqual(decide(ready(state=4, last_run=800), [], {}, 1000)["action"], "restart")

    def test_explicit_pause_disable_handoff_and_manual_task_stop_are_respected(self):
        for task, blocked in ((ready(enabled=False), None), (ready(last_result=0x41306), None),
                              (ready(), "HANDOFF.json"), (ready(), "AUTO_RECOVERY_PAUSED")):
            result = decide(task, [], {}, 1000, blocked=blocked)
            self.assertEqual(result["state"], "paused")
            self.assertIsNone(result["action"])

    def test_monitor_requires_three_failed_health_checks_and_retries_are_bounded(self):
        result = {}
        for tick in (1000, 1060):
            result = decide(ready(state=4), [42], result, tick, web_ok=False)
            self.assertIsNone(result["action"])
        result = decide(ready(state=4), [42], result, 1120, web_ok=False)
        self.assertEqual(result["action"], "restart")
        wait = decide(ready(), [], result, 1140)
        self.assertEqual(wait["state"], "retry_wait")
        self.assertIsNone(wait["action"])
        self.assertEqual(decide(ready(), [], wait, 1181)["action"], "run")

    def test_read_only_cli_and_other_projects_do_not_count_as_training(self):
        root = Path("C:/GoGoGo")
        rows = [{"pid": 1, "command": f'"python.exe" "{root / "train_rl.py"}" curriculum --hardware auto'},
                {"pid": 2, "command": f'python.exe "{root / "train_rl.py"}" curriculum-status'},
                {"pid": 3, "command": 'python.exe C:/other/train_rl.py curriculum'}]
        self.assertEqual(matching_pids(root, rows, "training"), [1])


class WatchdogPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "config").mkdir()
        self.state = self.root / "training_runs" / "curriculum"
        self.state.mkdir(parents=True)
        (self.root / "config" / "rl_curriculum.rtx5070ti.json").write_text(json.dumps({
            "state_directory": "training_runs/curriculum", "stages": [{"board_size": 19, "training_config": "config/stage.json"}]}))
        (self.root / "config" / "stage.json").write_text(json.dumps({"overrides": {"runtime": {"output_directory": "training_runs/19x19"}}}))
        (self.state / "state.json").write_text(json.dumps({"active_board_size": 19, "phase": "evaluating"}))

    def test_stage_handoff_and_manual_pause_prevent_restarting(self):
        stage = self.root / "training_runs" / "19x19"
        stage.mkdir()
        marker = stage / "HANDOFF.json"
        marker.write_text("{}")
        self.assertIn("锁定", training_block(self.root))
        marker.unlink()
        (self.state / "AUTO_RECOVERY_PAUSED").write_text("manual")
        self.assertIn("主动暂停", training_block(self.root))

    def test_failure_is_persisted_then_success_is_recorded_without_training_writes(self):
        before = (self.state / "state.json").read_bytes()
        snapshot = {"tasks": {"training": ready()}, "processes": []}
        def failure(_name, _action):
            raise RuntimeError("task start rejected")
        result = check(self.root, tasks={"training": "GoGoGo-Test"}, snapshot=snapshot,
                       action_runner=failure, web_ok=True, now=1000)
        self.assertEqual(result["training"]["state"], "failed")
        self.assertIn("task start rejected", result["events"][-1]["message"])
        persisted = json.loads((self.state / "recovery_status.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["training"]["restart_count"], 1)
        actions = []
        check(self.root, tasks={"training": "GoGoGo-Test"}, snapshot=snapshot,
              action_runner=lambda *args: actions.append(args), web_ok=True, now=1020)
        self.assertEqual(actions, [])
        snapshot["processes"] = [{"pid": 99, "command": f'python "{self.root / "train_rl.py"}" curriculum'}]
        healthy = check(self.root, tasks={"training": "GoGoGo-Test"}, snapshot=snapshot,
                        action_runner=lambda *args: actions.append(args), web_ok=True, now=1100)
        self.assertEqual(healthy["training"]["state"], "healthy")
        self.assertIn("恢复完成", healthy["events"][-1]["message"])
        self.assertEqual((self.state / "state.json").read_bytes(), before)
