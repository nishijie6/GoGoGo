"""GPU query failures and caching must not interrupt training monitoring."""

import subprocess
from datetime import datetime, timedelta, timezone
import threading
import unittest
from unittest.mock import patch

from weiqi.gpu_monitor import GPUMonitor


class GPUMonitorTests(unittest.TestCase):
    @patch("weiqi.gpu_monitor.subprocess.run")
    def test_multiple_gpus_missing_metrics_and_idle_zero(self, run):
        run.return_value.stdout = '0, "GPU, A", 0, 1024, 16384, 65\n1, GPU B, [N/A], 512, 8192, [N/A]\n'
        result = GPUMonitor().snapshot()
        self.assertTrue(result["available"])
        self.assertIsNotNone(result["sampled_at"])
        self.assertEqual(result["gpus"][0]["name"], "GPU, A")
        self.assertEqual(result["gpus"][0]["utilization_percent"], 0)
        self.assertIsNone(result["gpus"][1]["utilization_percent"])
        self.assertIsNone(result["gpus"][1]["temperature_c"])
        self.assertEqual(run.call_args.kwargs["timeout"], 2)

    @patch("weiqi.gpu_monitor.time.monotonic")
    @patch("weiqi.gpu_monitor.subprocess.run")
    def test_cache_refresh_and_failure_clear_previous_reading(self, run, monotonic):
        monotonic.return_value = 10
        run.return_value.stdout = '0, GPU, 95, 2048, 16384, 65\n'
        monitor = GPUMonitor()
        self.assertTrue(monitor.snapshot()["available"])
        self.assertEqual(monitor.snapshot()["gpus"][0]["utilization_percent"], 95)
        self.assertEqual(len(monitor.snapshot()["history"]), 1)
        run.assert_called_once()
        monotonic.return_value = 13
        run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 2)
        result = monitor.snapshot()
        self.assertFalse(result["available"])
        self.assertEqual(result["gpus"], [])
        self.assertIsNone(result["sampled_at"])
        self.assertIn("超时", result["reason"])
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(result["history"][0]["gpus"][0]["utilization_percent"], 95)
        self.assertEqual(result["history"][1]["gpus"], [])

    @patch("weiqi.gpu_monitor.subprocess.run")
    def test_history_is_bounded_and_expires_after_thirty_minutes(self, run):
        run.return_value.stdout = '0, GPU, 95, 2048, 16384, 65\n'
        monitor = GPUMonitor()
        initial = datetime.now(timezone.utc)
        with patch("weiqi.gpu_monitor.datetime") as clock:
            clock.now.return_value = initial
            clock.fromisoformat.side_effect = datetime.fromisoformat
            for _ in range(905):
                monitor.expires = 0
                result = monitor.snapshot()
            self.assertEqual(len(result["history"]), 900)
            clock.now.return_value = initial + timedelta(minutes=31)
            monitor.expires = 0
            self.assertEqual(len(monitor.snapshot()["history"]), 1)

    def test_background_sampling_runs_without_browser_and_stops(self):
        monitor = GPUMonitor()
        collected = threading.Event()
        with patch.object(monitor, "snapshot", side_effect=collected.set) as sample:
            try:
                monitor.start()
                self.assertTrue(collected.wait(timeout=1))
                monitor.start()
            finally:
                monitor.stop()
            self.assertFalse(monitor.worker.is_alive())
            sample.assert_called_once()

    def test_missing_driver_failed_command_and_malformed_output(self):
        for error in (FileNotFoundError(), subprocess.CalledProcessError(1, "nvidia-smi")):
            with self.subTest(error=error), patch("weiqi.gpu_monitor.subprocess.run", side_effect=error):
                self.assertFalse(GPUMonitor().snapshot()["available"])
        with patch("weiqi.gpu_monitor.subprocess.run") as run:
            run.return_value.stdout = 'bad output\n0, GPU, NaN, -1, 16000, Infinity\n'
            gpu = GPUMonitor().snapshot()["gpus"][0]
            self.assertIsNone(gpu["utilization_percent"])
            self.assertIsNone(gpu["memory_used_mib"])
            self.assertIsNone(gpu["temperature_c"])
