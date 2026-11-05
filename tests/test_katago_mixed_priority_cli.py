"""Diagnostic/evaluation commands must not accidentally start a training loop."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weiqi.katago_hardware import HardwareProfile
from weiqi.katago_rl_cli import main


class MixedPriorityCLITests(unittest.TestCase):
    def test_budget_candidates_use_only_frozen_snapshot_without_gpu_or_training_runner(self):
        with patch("weiqi.katago_rl_cli.MixedBoardRunner") as factory, \
                patch("weiqi.katago_rl_cli.prepare_budget_candidates", return_value={"status": "prepared"}) as prepare, \
                patch("weiqi.katago_rl_cli.detect_hardware", side_effect=AssertionError("GPU probe")):
            main(["mixed-budget-prepare", "--experiment-output", "/tmp/base", "--destination", "/tmp/candidates"])
        factory.assert_not_called()
        prepare.assert_called_once_with(Path("/tmp/base"), Path("/tmp/candidates"))

    def test_preparation_is_cpu_only_and_takes_the_source_lock(self):
        with patch("weiqi.katago_rl_cli.MixedBoardRunner") as factory, \
                patch("weiqi.katago_rl_cli.prepare_mixed_experiment", return_value={"status": "prepared"}) as prepare, \
                patch("weiqi.katago_rl_cli.detect_hardware", side_effect=AssertionError("GPU probe")):
            main(["mixed-experiment-prepare", "--experiment-output", "/tmp/trial", "--budget-seconds", "180"])
        factory.return_value.lock.assert_called_once()
        factory.return_value.initialize_model.assert_not_called()
        prepare.assert_called_once()

    def test_experiment_run_requires_output_before_gpu_probe(self):
        with patch("weiqi.katago_rl_cli.detect_hardware", side_effect=AssertionError("GPU probe")):
            with self.assertRaises(SystemExit) as error:
                main(["mixed-experiment-run"])
        self.assertEqual(error.exception.code, 2)

    def test_health_is_cpu_only_and_does_not_take_training_lock(self):
        with patch("weiqi.katago_rl_cli.MixedBoardRunner") as factory, \
                patch("weiqi.katago_rl_cli.build_mixed_health", return_value={"status":"ok"}) as health, \
                patch("weiqi.katago_rl_cli.detect_hardware", side_effect=AssertionError("GPU probe")):
            main(["mixed-health"])
        factory.return_value.initialize_model.assert_not_called()
        factory.return_value.cycle.assert_not_called()
        factory.return_value.lock.assert_not_called()
        health.assert_called_once()

    def test_evaluation_requires_explicit_protocol_and_never_trains(self):
        profile=HardwareProfile("cpu","cpu","float32",None,0,1,False)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"eval";baseline=Path(tmp)/"baseline.bin.gz"
            with patch("weiqi.katago_rl_cli.MixedBoardRunner") as factory, \
                    patch("weiqi.katago_rl_cli.evaluate_mixed", return_value={"complete":True}) as evaluate, \
                    patch("weiqi.katago_rl_cli.detect_hardware",return_value=profile):
                main(["mixed-evaluate","--baseline-model",str(baseline),
                      "--evaluation-output",str(out),"--evaluation-games","12",
                      "--evaluation-visits","32","--evaluation-boards","7","8"])
            factory.return_value.cycle.assert_not_called()
            factory.return_value.initialize_model.assert_not_called()
            self.assertEqual(evaluate.call_args.kwargs["games_per_board"],12)
            self.assertEqual(evaluate.call_args.kwargs["boards"],[7,8])
            self.assertEqual(evaluate.call_args.kwargs["baseline"],baseline)
            self.assertEqual(evaluate.call_args.args[1],out)

    def test_missing_baseline_fails_before_device_discovery(self):
        with patch("weiqi.katago_rl_cli.detect_hardware",side_effect=AssertionError("GPU probe")), \
                self.assertRaises(SystemExit) as error:
            main(["mixed-evaluate","--evaluation-output","unused"])
        self.assertEqual(error.exception.code,2)


if __name__ == "__main__":
    unittest.main()
