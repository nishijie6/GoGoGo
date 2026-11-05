"""The official trainer must respect an explicit hardware selection."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class KataGoTrainTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("weiqi.katago_train"),
                             "The explicit-device training launcher is missing")
        self.launcher = importlib.import_module("weiqi.katago_train")
        import torch
        self.torch = torch
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.script = Path(self.temporary.name) / "train.py"

    def write_script(self, text):
        self.script.write_text(text, encoding="utf-8")

    def test_cpu_hides_available_cuda_and_mps_and_preserves_official_arguments(self):
        self.write_script("""
import torch, sys
assert not torch.cuda.is_available()
assert not torch.backends.mps.is_available()
assert torch.cuda.device_count() == 0
assert sys.argv[1:] == ['-pos-len', '19', '--memory-fraction', 'official-value']
""")
        with patch.object(self.torch.cuda, "is_available", return_value=True), \
             patch.object(self.torch.backends.mps, "is_available", return_value=True):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.launcher.run_training(self.script, "cpu", ["-pos-len", "19", "--memory-fraction", "official-value"])
            self.assertTrue(self.torch.cuda.is_available())
            self.assertTrue(self.torch.backends.mps.is_available())
        self.assertIn('KATAGO_TRAIN_DEVICE={"device": "cpu"}', output.getvalue())

    def test_restores_argv_path_and_discovery_when_script_raises(self):
        self.write_script("raise RuntimeError('official failure')")
        argv, path = sys.argv, list(sys.path)
        cuda_available = self.torch.cuda.is_available
        mps_available = self.torch.backends.mps.is_available
        with self.assertRaisesRegex(RuntimeError, "official failure"):
            self.launcher.run_training(self.script, "cpu", ["-a", "value"])
        self.assertIs(sys.argv, argv)
        self.assertEqual(sys.path, path)
        self.assertIs(self.torch.cuda.is_available, cuda_available)
        self.assertIs(self.torch.backends.mps.is_available, mps_available)

    def test_restores_path_object_if_official_script_replaces_it(self):
        self.write_script("import sys; sys.path = ['replaced']; raise RuntimeError('failure')")
        original_path = sys.path
        original_entries = list(sys.path)
        try:
            with self.assertRaisesRegex(RuntimeError, "failure"):
                self.launcher.run_training(self.script, "cpu", [])
            self.assertIs(sys.path, original_path)
            self.assertEqual(sys.path, original_entries)
        finally:
            sys.path = original_path
            sys.path[:] = original_entries

    def test_first_model_training_forward_checks_actual_input_device(self):
        self.write_script("""
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        raise AssertionError('forward should not execute on the wrong device')
Model.__module__ = 'katago.train.model_pytorch'
Model()(torch.ones(1, device='meta'))
""")
        with self.assertRaisesRegex(RuntimeError, "expected cpu.*meta"):
            self.launcher.run_training(self.script, "cpu", [])

    def test_forward_check_is_removed_after_first_verified_training_batch(self):
        self.write_script("""
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return x
Model.__module__ = 'katago.train.model_pytorch'
m = Model()
m(torch.ones(1))
m(torch.ones(1, device='meta'))
""")
        hooks = self.torch.nn.modules.module._global_forward_pre_hooks
        before = dict(hooks)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.launcher.run_training(self.script, "cpu", [])
        self.assertEqual(dict(hooks), before)
        self.assertIn('KATAGO_TRAIN_FORWARD_DEVICE={"device": "cpu"}', output.getvalue())

    def test_unavailable_requested_accelerator_fails_instead_of_running_script(self):
        self.write_script("raise AssertionError('must not start')")
        for device in ("mps", "cuda:0"):
            with self.subTest(device=device), \
                 patch.object(self.torch.cuda, "is_available", return_value=False), \
                 patch.object(self.torch.backends.mps, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    self.launcher.run_training(self.script, device, [])

    def test_cuda_index_and_official_device_argument_must_agree(self):
        self.write_script("raise AssertionError('must not start')")
        with patch.object(self.torch.cuda, "is_available", return_value=True), \
             patch.object(self.torch.cuda, "device_count", return_value=2):
            with self.assertRaisesRegex(ValueError, "CUDA index"):
                self.launcher.run_training(self.script, "cuda:2", [])
            with self.assertRaisesRegex(ValueError, "multi-gpus"):
                self.launcher.run_training(self.script, "cuda:1", ["-multi-gpus", "0"])

    def test_cuda_selection_hides_mps_without_altering_official_arguments(self):
        self.write_script("""
import torch, sys
assert torch.cuda.is_available()
assert not torch.backends.mps.is_available()
assert sys.argv[1:] == ['-multi-gpus', '1']
""")
        with patch.object(self.torch.cuda, "is_available", return_value=True), \
             patch.object(self.torch.cuda, "device_count", return_value=2), \
             patch.object(self.torch.backends.mps, "is_available", return_value=True):
            self.launcher.run_training(self.script, "cuda:1", ["-multi-gpus", "1"])

    def test_mps_sets_limit_disables_cpu_fallback_and_executes_kernel(self):
        self.write_script("""
import os, torch
assert not torch.cuda.is_available()
assert torch.backends.mps.is_available()
assert os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] == '0'
""")
        state = {"limit": None, "kernel": False}
        class Probe:
            def __mul__(self, other):
                state["kernel"] = True
                return self
            def sum(self):
                return self
            def item(self):
                if not state["kernel"]:
                    raise AssertionError("No kernel executed")
                return 4.0
        def allocate(*args, **kwargs):
            self.assertEqual(kwargs["device"], "mps")
            self.assertEqual(state["limit"], .45)
            return Probe()
        def limit(value):
            state["limit"] = value
        with patch.object(self.torch.backends.mps, "is_available", return_value=True), \
             patch.object(self.torch.mps, "set_per_process_memory_fraction", side_effect=limit), \
             patch.object(self.torch, "ones", side_effect=allocate):
            self.launcher.run_training(self.script, "mps", [], memory_fraction=.45)
        self.assertTrue(state["kernel"])

    def test_mps_kernel_failure_is_not_silently_replaced_by_cpu(self):
        self.write_script("raise AssertionError('must not start')")
        with patch.object(self.torch.backends.mps, "is_available", return_value=True), \
             patch.object(self.torch.mps, "set_per_process_memory_fraction"), \
             patch.object(self.torch, "ones", side_effect=RuntimeError("MPS broken")):
            with self.assertRaisesRegex(RuntimeError, "MPS broken"):
                self.launcher.run_training(self.script, "mps", [])

    def test_invalid_device_and_memory_fraction_fail_before_importing_torch(self):
        self.write_script("raise AssertionError('must not start')")
        for device in ("auto", "cuda", "cuda:-1", "mps:1"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                self.launcher.run_training(self.script, device, [])
        for fraction in (0, -.1, 1.1, float('nan'), float('inf')):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                self.launcher.run_training(self.script, "cpu", [], fraction)

    def test_cli_runs_cpu_in_launcher_interpreter_and_preserves_remainder(self):
        output = Path(self.temporary.name) / "result.json"
        self.write_script(f"""
import json, sys
from pathlib import Path
Path({str(output)!r}).write_text(json.dumps({{'executable': sys.executable, 'argv': sys.argv}}))
""")
        completed = subprocess.run(
            [sys.executable, str(Path(self.launcher.__file__)), "--device", "cpu",
             "--memory-fraction", ".5", str(self.script), "-pos-len", "19", "--device", "official"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(output.read_text())
        self.assertEqual(result["executable"], sys.executable)
        self.assertEqual(result["argv"], [str(self.script), "-pos-len", "19", "--device", "official"])


if __name__ == "__main__":
    unittest.main()
