"""Ordinary trainer relay commands remain separate from KataGo and training imports."""

from __future__ import annotations

import builtins
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import train as cli
from weiqi import rl_handoff
from weiqi.rl_config import resolve_rl_training_config


class NativeHandoffCLITests(unittest.TestCase):
    def run_command(self, argv: list[str], api_name: str, report: dict):
        output = io.StringIO()
        with patch.object(rl_handoff, api_name, return_value=report) as api, contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(argv), 0)
        return api, json.loads(output.getvalue())

    def test_export_requires_existing_run_selection(self) -> None:
        api, printed = self.run_command(
            ["handoff-export", "--run-dir", "training_runs/balanced", "--bundle", "one.zip"],
            "export_handoff", {"trainer": "native", "files": 4},
        )
        api.assert_called_once_with(cli.ROOT, Path("one.zip"), native_run=Path("training_runs/balanced"))
        self.assertEqual(printed["trainer"], "native")

    def test_verify_declares_native_backend(self) -> None:
        api, _ = self.run_command(["handoff-verify", "--bundle", "one.zip"], "verify_handoff", {"files": 3})
        api.assert_called_once_with(Path("one.zip"), expected_trainer="native")

    def test_import_outputs_checkpoint_based_resume_command_and_preserves_api_report(self) -> None:
        report = {"trainer": "native", "resume_checkpoint": Path("local relay/run/latest.pt"),
                  "training_config": Path("local relay/config/training.json"),
                  "hardware_profile": Path("local relay/config/hardware.json")}
        api, printed = self.run_command(
            ["handoff-import", "--bundle", "one.zip", "--destination", "training_runs/relay",
             "--hardware-profile", "machine.json", "--allow-version-mismatch"],
            "import_handoff", report,
        )
        api.assert_called_once_with(
            cli.ROOT, Path("one.zip"), Path("training_runs/relay"),
            allow_version_mismatch=True, hardware_profile=Path("machine.json"), expected_trainer="native",
        )
        command = printed["resume_command"]
        self.assertIn("train.py", command)
        self.assertIn("--resume", command)
        self.assertIn(str(report["resume_checkpoint"]), command)
        self.assertIn("--iterations", command)
        self.assertIn("--hardware-overrides", command)
        self.assertNotIn("--config", command)
        self.assertNotIn("resume_command", report)

    def test_import_without_hardware_profile_uses_saved_checkpoint_settings(self) -> None:
        _, printed = self.run_command(
            ["handoff-import", "--bundle", "one.zip", "--destination", "new"],
            "import_handoff", {"resume_checkpoint": "new/run/latest.pt"},
        )
        self.assertNotIn("--hardware-overrides", printed["resume_command"])

    def test_windows_resume_paths_are_quoted_as_literal_arguments(self) -> None:
        with patch.object(cli.sys, "platform", "win32"):
            command = cli._resume_command({"resume_checkpoint": "a's $(literal)/latest.pt"})
        self.assertTrue(command.startswith("& "))
        self.assertIn("'a''s $(literal)/latest.pt'", command)

    def test_posix_resume_command_preserves_argument_boundaries(self) -> None:
        import shlex
        with patch.object(cli.sys, "platform", "linux"):
            command = cli._resume_command({"resume_checkpoint": "a's $(literal)/latest.pt"})
        self.assertEqual(shlex.split(command)[-4:], ["--resume", "a's $(literal)/latest.pt", "--iterations", "1"])

    def test_check_declares_backend_and_only_loads_on_request(self) -> None:
        for flag in ([], ["--load-checkpoint"]):
            with self.subTest(flag=flag):
                api, _ = self.run_command(
                    ["handoff-check", "--destination", "new", *flag], "check_import", {"files": 3},
                )
                api.assert_called_once_with(Path("new"), load_checkpoint=bool(flag), expected_trainer="native")

    def test_cancel_declares_native_run(self) -> None:
        api, _ = self.run_command(
            ["handoff-cancel", "--run-dir", "training_runs/balanced"], "cancel_handoff", {"removed": 1},
        )
        api.assert_called_once_with(cli.ROOT, native_run=Path("training_runs/balanced"))

    def test_missing_flags_and_backend_specific_options_are_usage_errors(self) -> None:
        invalid = (
            ["handoff-export", "--bundle", "one.zip"],
            ["handoff-export", "--run-dir", "source"], ["handoff-verify"],
            ["handoff-import", "--bundle", "one.zip"], ["handoff-import", "--destination", "new"],
            ["handoff-check"], ["handoff-cancel"],
            ["handoff-export", "--run-dir", "source", "--bundle", "one.zip", "--single-run"],
            ["handoff-verify", "--bundle", "one.zip", "--load-checkpoint"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.main(argv)
                self.assertEqual(error.exception.code, 2)

    def test_wrong_backend_is_a_nonzero_actionable_failure(self) -> None:
        errors = io.StringIO()
        with (patch.object(rl_handoff, "verify_handoff", side_effect=rl_handoff.HandoffError("Expected native, got katago")),
              contextlib.redirect_stderr(errors)):
            self.assertEqual(cli.main(["handoff-verify", "--bundle", "katago.zip"]), 1)
        self.assertIn("katago", errors.getvalue())

    def test_relay_dispatch_does_not_import_training_stack(self) -> None:
        original = builtins.__import__

        def reject_training(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch.") or name.startswith("weiqi.rl."):
                raise AssertionError(f"Unexpected training import: {name}")
            return original(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=reject_training):
            self.run_command(["handoff-verify", "--bundle", "one.zip"], "verify_handoff", {"files": 3})
            self.run_command(["handoff-import", "--bundle", "one.zip", "--destination", "new"],
                             "import_handoff", {"resume_checkpoint": "new/run/latest.pt"})
            self.run_command(["handoff-check", "--destination", "new"], "check_import", {"files": 3})

    def test_hardware_overrides_apply_after_saved_config_and_policy(self) -> None:
        # Substitute runtime modules: this exercises CLI resume wiring without
        # running games, mutating a checkpoint, or requiring torch in Windows.
        saved_config = resolve_rl_training_config("balanced")
        policy_config = replace(saved_config, optimizer=replace(saved_config.optimizer, batch_size=64))
        hardware_config = replace(policy_config, hardware=replace(policy_config.hardware, device="cpu"))
        payload = {"kind": "training", "iteration": 7}
        runner = ModuleType("weiqi.rl.runner")
        runner.train = Mock()
        control = ModuleType("weiqi.rl.control")
        control.TrainingStopped = type("TrainingStopped", (RuntimeError,), {})
        assessment = ModuleType("weiqi.rl.assessment")
        assessment.benchmark = Mock()
        assessment.evaluate = Mock()
        storage = ModuleType("weiqi.rl.storage")
        storage.fingerprint = Mock()
        storage.load_checkpoint = Mock(return_value=(payload, saved_config))
        native = ModuleType("weiqi.rl_native_handoff")
        native.apply_native_hardware = Mock(return_value=hardware_config)
        modules = {module.__name__: module for module in (runner, control, assessment, storage, native)}
        with (patch.dict(sys.modules, modules),
              patch("weiqi.fileio.sha256_file", return_value="checksum"),
              patch.object(cli, "with_training_policy", return_value=policy_config)):
            self.assertEqual(cli.main(["train", "--resume", "new/run/latest.pt", "--iterations", "1",
                                       "--policy-overrides", "policy.json", "--hardware-overrides", "hardware.json"]), 0)
        native.apply_native_hardware.assert_called_once_with(policy_config, Path("hardware.json"))
        passed_config, passed_output, iterations, passed_payload = runner.train.call_args.args
        self.assertEqual(passed_config.hardware.device, "cpu")
        self.assertEqual(passed_config.network, saved_config.network)
        self.assertEqual(passed_config.optimizer.batch_size, 64)
        self.assertEqual(passed_output, Path("new/run").resolve())
        self.assertEqual(iterations, 1)
        self.assertIs(passed_payload, payload)
        self.assertEqual(runner.train.call_args.kwargs["resume_checksum"], "checksum")


if __name__ == "__main__":
    unittest.main()
