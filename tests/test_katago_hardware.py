"""Hardware adaptation must work without a GPU and preserve training intent."""

from __future__ import annotations

import contextlib
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from weiqi import katago_hardware
from weiqi.katago_hardware import HardwareProfile, detect_hardware
from weiqi.rl_config import resolve_rl_training_config


class _ProbeTensor:
    def __init__(self, error: str | None) -> None:
        self.error = error
        self.multiplied = False

    def __mul__(self, other):
        self.multiplied = True
        return self

    def sum(self):
        return self

    def item(self):
        if self.error:
            raise RuntimeError(self.error)
        if not self.multiplied:
            raise AssertionError("The CUDA check must execute a tensor kernel")
        return 4.0


class _Cuda:
    def __init__(self, specs, *, available=True) -> None:
        self.specs = specs
        self.available = available
        self.current_index = 0
        self.bf16_probes = []

    def is_available(self):
        return self.available

    def device_count(self):
        return len(self.specs)

    def get_device_properties(self, index):
        memory, capability, bf16, error = self.specs[index]
        return SimpleNamespace(total_memory=memory * 1024**3, name=f"GPU {index}")

    def get_device_capability(self, index):
        return self.specs[index][1]

    @contextlib.contextmanager
    def device(self, index):
        previous = self.current_index
        self.current_index = index
        try:
            yield
        finally:
            self.current_index = previous

    def is_bf16_supported(self):
        self.bf16_probes.append(self.current_index)
        return self.specs[self.current_index][2]


def _fake_torch(*specs, available=True):
    # Each device: memory GiB, compute capability, BF16 support, kernel error.
    cuda = _Cuda(specs, available=available)
    tensors = []

    def ones(shape, *, dtype, device):
        index = int(device.partition(":")[2])
        tensor = _ProbeTensor(specs[index][3])
        tensors.append((device, tensor))
        return tensor

    return SimpleNamespace(cuda=cuda, ones=ones, float32="float32", tensors=tensors)


def _profile(tier="cpu", *, memory=0.0, cpu_threads=12):
    return HardwareProfile(
        tier=tier,
        device="cpu" if tier == "cpu" else "cuda:1",
        precision="float32" if tier == "cpu" else "amp_bfloat16",
        gpu_name=None if tier == "cpu" else "Test GPU",
        memory_gib=memory,
        cpu_threads=cpu_threads,
        use_fp16=tier != "cpu",
    )


class DetectHardwareTests(unittest.TestCase):
    def _detect(self, torch, tier="auto"):
        with patch.object(katago_hardware.importlib, "import_module", return_value=torch), \
                patch.object(katago_hardware.os, "cpu_count", return_value=12):
            return detect_hardware(tier)

    def test_forced_cpu_never_imports_torch_even_when_cuda_exists(self):
        with patch.object(katago_hardware.importlib, "import_module") as load, \
                patch.object(katago_hardware.os, "cpu_count", return_value=None):
            profile = detect_hardware("cpu")
        load.assert_not_called()
        self.assertEqual(profile.tier, "cpu")
        self.assertEqual(profile.precision, "float32")
        self.assertEqual(profile.cpu_threads, 1)
        self.assertFalse(profile.use_fp16)

    def test_missing_torch_falls_back_and_records_reason(self):
        with patch.object(katago_hardware.importlib, "import_module",
                   side_effect=ImportError("No module named torch")):
            profile = detect_hardware()
        self.assertEqual(profile.device, "cpu")
        self.assertIn("No module named torch", profile.reason)

    def test_broken_torch_dll_falls_back_to_cpu(self):
        with patch.object(katago_hardware.importlib, "import_module",
                   side_effect=OSError("DLL load failed")):
            profile = detect_hardware()
        self.assertEqual(profile.tier, "cpu")
        self.assertIn("DLL load failed", profile.reason)

    def test_unavailable_cuda_falls_back_to_cpu(self):
        profile = self._detect(_fake_torch(available=False))
        self.assertEqual(profile.backend, "eigen")
        self.assertIn("CUDA", profile.reason)

    def test_driver_initialization_failure_falls_back(self):
        torch = _fake_torch()
        torch.cuda.device_count = lambda: (_ for _ in ()).throw(ValueError("driver failure"))
        profile = self._detect(torch)
        self.assertEqual(profile.tier, "cpu")
        self.assertIn("driver failure", profile.reason)

    def test_explicit_gpu_tiers_fail_instead_of_falling_back(self):
        for tier in ("low", "high"):
            with self.subTest(tier=tier), self.assertRaisesRegex(ValueError, "需要可用的 CUDA"):
                self._detect(_fake_torch(available=False), tier)
            with self.subTest(tier=tier, missing_torch=True), \
                    patch.object(katago_hardware.importlib, "import_module",
                          side_effect=ImportError("missing torch")), \
                    self.assertRaisesRegex(ValueError, "需要可用的 CUDA"):
                detect_hardware(tier)

    def test_invalid_tier_fails_before_loading_torch(self):
        with patch.object(katago_hardware.importlib, "import_module") as load, \
                self.assertRaisesRegex(ValueError, "未知硬件档位"):
            detect_hardware("medium")
        load.assert_not_called()

    def test_high_requires_both_memory_and_modern_compute_capability(self):
        for memory, capability, expected in (
            (12, (8, 0), "high"),
            (11.99, (9, 0), "low"),
            (24, (7, 5), "low"),
            (16, (12, 0), "high"),
        ):
            with self.subTest(memory=memory, capability=capability):
                profile = self._detect(_fake_torch((memory, capability, True, None)))
                self.assertEqual(profile.tier, expected)

    def test_highest_memory_working_gpu_is_selected_and_bf16_probed_on_it(self):
        torch = _fake_torch((8, (8, 6), False, None), (16, (8, 9), True, None))
        profile = self._detect(torch)
        self.assertEqual(profile.device, "cuda:1")
        self.assertEqual(profile.gpu_index, 1)
        self.assertEqual(profile.gpu_name, "GPU 1")
        self.assertEqual(profile.memory_gib, 16)
        self.assertEqual(profile.precision, "amp_bfloat16")
        self.assertEqual(torch.cuda.bf16_probes, [1])
        self.assertEqual(torch.cuda.current_index, 0)
        self.assertTrue(all(tensor.multiplied for _, tensor in torch.tensors))

    def test_unsupported_architecture_is_skipped_for_working_smaller_gpu(self):
        torch = _fake_torch(
            (24, (12, 0), True, "no kernel image is available for execution"),
            (6, (7, 5), False, None),
        )
        profile = self._detect(torch)
        self.assertEqual(profile.device, "cuda:1")
        self.assertEqual(profile.tier, "low")
        self.assertIn("no kernel image", profile.reason)

    def test_all_gpu_kernel_failures_fall_back_or_raise_for_explicit_tier(self):
        torch = _fake_torch((16, (12, 0), True, "unsupported architecture"))
        profile = self._detect(torch)
        self.assertEqual(profile.tier, "cpu")
        self.assertIn("unsupported architecture", profile.reason)
        with self.assertRaisesRegex(ValueError, "unsupported architecture"):
            self._detect(torch, "high")

    def test_precision_falls_back_to_fp16_or_float32(self):
        for capability, bf16, expected, fp16 in (
            ((8, 0), False, "amp_float16", True),
            ((7, 5), True, "amp_float16", True),
            ((6, 1), True, "float32", False),
        ):
            with self.subTest(capability=capability):
                profile = self._detect(_fake_torch((8, capability, bf16, None)))
                self.assertEqual(profile.precision, expected)
                self.assertEqual(profile.use_fp16, fp16)

    def test_bf16_probe_error_does_not_discard_working_gpu(self):
        torch = _fake_torch((16, (8, 6), True, None))
        torch.cuda.is_bf16_supported = lambda: (_ for _ in ()).throw(RuntimeError("unsupported"))
        profile = self._detect(torch)
        self.assertEqual(profile.tier, "high")
        self.assertEqual(profile.precision, "amp_float16")

    def test_explicit_tier_overrides_auto_classification(self):
        self.assertEqual(self._detect(_fake_torch((16, (8, 6), True, None)), "low").tier, "low")
        self.assertEqual(self._detect(_fake_torch((8, (8, 6), True, None)), "high").tier, "high")


class HardwareProfileTests(unittest.TestCase):
    def test_board_size_batch_limits_cover_cpu_small_gpu_and_large_gpu(self):
        for profile, expected in (
            (_profile("cpu"), (16, 8, 4)),
            (_profile("low", memory=4), (64, 32, 16)),
            (_profile("low", memory=4.01), (256, 128, 64)),
            (_profile("high", memory=16), (1024, 2048, 1280)),
        ):
            with self.subTest(tier=profile.tier, memory=profile.memory_gib):
                self.assertEqual(tuple(profile.training_batch_limit(size) for size in (9, 13, 19)), expected)
        with self.assertRaisesRegex(ValueError, "只支持"):
            _profile().training_batch_limit(5)

    def test_apply_cpu_caps_throughput_and_preserves_model_training_objective(self):
        original = resolve_rl_training_config("high_performance")
        adapted = _profile().apply(original)
        self.assertEqual(adapted.hardware.device, "cpu")
        self.assertEqual(adapted.hardware.precision, "float32")
        self.assertFalse(adapted.hardware.allow_tf32)
        self.assertFalse(adapted.hardware.compile_model)
        self.assertFalse(adapted.hardware.pin_memory)
        self.assertEqual(adapted.hardware.data_loader_workers, 0)
        self.assertEqual(adapted.self_play.workers, 1)
        self.assertEqual(adapted.self_play.games_per_iteration, 4)
        self.assertEqual(adapted.self_play.inference_batch_size, 1)
        self.assertEqual(adapted.optimizer.batch_size, 4)
        for field in ("game", "network", "search", "evaluation", "runtime"):
            self.assertIs(getattr(adapted, field), getattr(original, field))
        self.assertEqual(adapted.optimizer.training_steps_per_iteration,
                         original.optimizer.training_steps_per_iteration)
        self.assertEqual(original.hardware.device, "auto")

    def test_apply_low_caps_only_resources_and_keeps_smaller_requested_values(self):
        original = resolve_rl_training_config("high_performance")
        adapted = _profile("low", memory=4).apply(original)
        self.assertEqual(adapted.hardware.device, "cuda:1")
        self.assertEqual(adapted.self_play.workers, 2)
        self.assertEqual(adapted.self_play.games_per_iteration, 32)
        self.assertEqual(adapted.self_play.inference_batch_size, 16)
        self.assertEqual(adapted.optimizer.batch_size, 16)
        self.assertFalse(adapted.hardware.compile_model)
        self.assertEqual(adapted.hardware.data_loader_workers, 2)
        self.assertIs(adapted.network, original.network)
        self.assertIs(adapted.search, original.search)
        self.assertIs(adapted.evaluation, original.evaluation)
        small = replace(original, self_play=replace(original.self_play, workers=1,
                        games_per_iteration=2, inference_batch_size=1),
                        optimizer=replace(original.optimizer, batch_size=2))
        kept = _profile("low", memory=4).apply(small)
        self.assertEqual(kept.self_play, small.self_play)
        self.assertEqual(kept.optimizer.batch_size, 2)

    def test_apply_high_retains_existing_performance_configuration(self):
        original = resolve_rl_training_config("high_performance")
        profile = _profile("high", memory=16)
        adapted = profile.apply(original)
        self.assertEqual(adapted.hardware, replace(original.hardware, device=profile.device,
                                                  precision=profile.precision))
        self.assertIs(adapted.self_play, original.self_play)
        self.assertEqual(adapted.optimizer, original.optimizer)
        oversized = replace(original, optimizer=replace(original.optimizer, batch_size=2048))
        self.assertEqual(profile.apply(oversized).optimizer.batch_size, 1280)

    def test_backend_and_inference_settings_follow_selected_device(self):
        cpu = _profile()
        self.assertEqual(cpu.backend, "eigen")
        self.assertEqual(cpu.game_threads, 4)
        self.assertIsNone(cpu.gpu_index)
        overrides = cpu.inference_overrides()
        self.assertEqual(overrides["numEigenThreadsPerModel"], 12)
        self.assertEqual(overrides["nnCacheSizePowerOfTwo"], 16)
        self.assertEqual(overrides["numNNServerThreadsPerModel"], 1)
        self.assertFalse(any(key.startswith("cuda") for key in overrides))
        self.assertEqual(_profile(cpu_threads=1).game_threads, 1)
        low = _profile("low", memory=8)
        self.assertEqual(low.backend, "cuda")
        self.assertEqual(low.game_threads, 32)
        self.assertEqual(low.inference_overrides()["cudaDeviceToUse"], 1)
        self.assertTrue(low.inference_overrides()["cudaUseNHWC"])
        self.assertNotIn("cudaUseNHWC", low.inference_overrides(has_model=False))
        high = _profile("high", memory=16)
        self.assertIsNone(high.game_threads)
        self.assertEqual(high.inference_overrides()["nnCacheSizePowerOfTwo"], 20)
        self.assertEqual(high.inference_overrides(smoke=True)["nnCacheSizePowerOfTwo"], 18)
        self.assertEqual(high.inference_overrides(smoke=True)["numNNServerThreadsPerModel"], 1)

    def test_float32_gpu_does_not_request_fp16_or_nhwc(self):
        old = replace(_profile("low", memory=4), use_fp16=False, precision="float32")
        overrides = old.inference_overrides()
        self.assertFalse(overrides["cudaUseFP16"])
        self.assertFalse(overrides["cudaUseNHWC"])
        self.assertNotIn("numEigenThreadsPerModel", overrides)

    def test_diagnostics_serialize_without_torch_or_dataclass_values(self):
        values = replace(_profile(), reason="CPU fallback").to_dict()
        self.assertEqual(values["tier"], "cpu")
        self.assertEqual(values["reason"], "CPU fallback")
        self.assertIsNone(values["gpu_name"])


if __name__ == "__main__":
    unittest.main()
