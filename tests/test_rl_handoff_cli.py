"""CLI contract tests for KataGo's local relay package operations."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from weiqi import katago_rl_cli as cli
from weiqi.rl_handoff import HandoffError


class HandoffCLITests(unittest.TestCase):
    def run_command(self, argv: list[str], api_name: str, report: dict):
        output = io.StringIO()
        with (
            patch.object(cli, api_name, return_value=report) as api,
            patch.object(cli, "KataGoRLRunner") as runner,
            patch.object(cli, "CurriculumRuntime") as curriculum,
            contextlib.redirect_stdout(output),
        ):
            cli.main(argv)
        runner.assert_not_called()
        curriculum.assert_not_called()
        return api, json.loads(output.getvalue())

    def test_export_defaults_to_curriculum_and_prints_json(self) -> None:
        report = {"status": "已封存", "files": 4}
        api, printed = self.run_command(
            ["handoff-export", "--bundle", "handoff_bundles/one.gogogo-handoff.zip"],
            "export_handoff",
            report,
        )
        api.assert_called_once_with(
            cli.PROJECT_ROOT,
            Path("handoff_bundles/one.gogogo-handoff.zip"),
            curriculum_config=cli.DEFAULT_CURRICULUM_PROFILE,
        )
        self.assertEqual(printed, report)

    def test_single_run_export_uses_selected_training_config(self) -> None:
        api, _ = self.run_command(
            ["handoff-export", "--bundle", "one.zip", "--single-run", "--config", "local.json"],
            "export_handoff",
            {"files": 2},
        )
        api.assert_called_once_with(
            cli.PROJECT_ROOT,
            Path("one.zip"),
            training_config=Path("local.json"),
        )

    def test_verify_only_reads_the_requested_bundle(self) -> None:
        api, _ = self.run_command(["handoff-verify", "--bundle", "one.zip"], "verify_handoff", {"files": 2})
        api.assert_called_once_with(Path("one.zip"), expected_trainer="katago")

    def test_import_passes_explicit_options_and_outputs_resume_command(self) -> None:
        report = {"curriculum_config": "local relay/curriculum.json", "warnings": ["版本差异"]}
        api, printed = self.run_command(
            [
                "handoff-import", "--bundle", "one.zip", "--destination", "training_runs/relay",
                "--allow-version-mismatch", "--hardware-profile", "hardware.json",
            ],
            "import_handoff",
            report,
        )
        api.assert_called_once_with(
            cli.PROJECT_ROOT,
            Path("one.zip"),
            Path("training_runs/relay"),
            allow_version_mismatch=True,
            hardware_profile=Path("hardware.json"),
            expected_trainer="katago",
        )
        self.assertIn("curriculum", printed["resume_command"])
        self.assertIn("--curriculum-config", printed["resume_command"])
        self.assertIn("local relay/curriculum.json", printed["resume_command"])
        self.assertEqual(printed["warnings"], ["版本差异"])
        self.assertNotIn("resume_command", report)

    def test_single_run_import_outputs_continuous_resume_command(self) -> None:
        _, printed = self.run_command(
            ["handoff-import", "--bundle", "one.zip", "--destination", "training_runs/relay"],
            "import_handoff",
            {"training_config": Path("local relay/training.json")},
        )
        self.assertIn("continuous", printed["resume_command"])
        self.assertIn("--config", printed["resume_command"])
        self.assertNotIn("--curriculum-config", printed["resume_command"])
        self.assertEqual(printed["training_config"], str(Path("local relay/training.json")))

    def test_windows_resume_command_quotes_paths_as_literals(self) -> None:
        with patch.object(cli.sys, "platform", "win32"):
            command = cli._resume_command({"training_config": "a's $(literal)/training.json"})
        self.assertTrue(command.startswith("& "))
        self.assertIn("'a''s $(literal)/training.json'", command)

    def test_check_default_does_not_load_checkpoint(self) -> None:
        api, _ = self.run_command(
            ["handoff-check", "--destination", "training_runs/relay"], "check_import", {"files": 2},
        )
        api.assert_called_once_with(Path("training_runs/relay"), load_checkpoint=False, expected_trainer="katago")

    def test_check_can_request_cpu_checkpoint_validation(self) -> None:
        api, _ = self.run_command(
            ["handoff-check", "--destination", "training_runs/relay", "--load-checkpoint"],
            "check_import", {"files": 2},
        )
        api.assert_called_once_with(Path("training_runs/relay"), load_checkpoint=True, expected_trainer="katago")

    def test_cancel_defaults_to_curriculum(self) -> None:
        api, _ = self.run_command(["handoff-cancel"], "cancel_handoff", {"removed": 1})
        api.assert_called_once_with(cli.PROJECT_ROOT, curriculum_config=cli.DEFAULT_CURRICULUM_PROFILE)

    def test_single_run_cancel_uses_selected_training_config(self) -> None:
        api, _ = self.run_command(
            ["handoff-cancel", "--single-run", "--config", "local.json"], "cancel_handoff", {"removed": 1},
        )
        api.assert_called_once_with(cli.PROJECT_ROOT, training_config=Path("local.json"))

    def test_missing_required_flags_and_misplaced_options_return_usage_error(self) -> None:
        invalid = (
            ["handoff-export"],
            ["handoff-verify"],
            ["handoff-import", "--destination", "new"],
            ["handoff-import", "--bundle", "one.zip"],
            ["handoff-check"],
            ["handoff-check", "--destination", "new", "--single-run"],
            ["handoff-verify", "--bundle", "one.zip", "--allow-version-mismatch"],
            ["handoff-export", "--bundle", "one.zip", "--hardware-profile", "hw.json"],
            ["handoff-export", "--bundle", "one.zip", "--load-checkpoint"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.main(argv)
                self.assertEqual(error.exception.code, 2)

    def test_handoff_failure_returns_actionable_error_and_nonzero_exit(self) -> None:
        errors = io.StringIO()
        with (
            patch.object(cli, "verify_handoff", side_effect=HandoffError("文件校验不匹配")),
            contextlib.redirect_stderr(errors),
            self.assertRaises(SystemExit) as error,
        ):
            cli.main(["handoff-verify", "--bundle", "one.zip"])
        self.assertEqual(error.exception.code, 1)
        self.assertIn("文件校验不匹配", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
