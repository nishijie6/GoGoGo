"""Tests for orphan-resistant subprocess ownership."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import weiqi.process_control as process_control
from weiqi.process_control import ManagedProcessError, managed_popen


class ManagedProcessTests(unittest.TestCase):
    def test_exception_terminates_live_child(self) -> None:
        process = None
        with self.assertRaises(RuntimeError):
            with managed_popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ) as process:
                raise RuntimeError("simulated parent interruption")
        assert process is not None
        self.assertIsNotNone(process.poll())

    @unittest.skipUnless(os.name == "nt", "Windows Job Object behavior")
    def test_hard_parent_exit_closes_kill_on_close_job(self) -> None:
        import ctypes

        code = (
            "import os,subprocess,sys\n"
            "from weiqi.process_control import managed_popen\n"
            "with managed_popen([sys.executable,'-c','import time; time.sleep(60)'],"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) as child:\n"
            " os.write(1,(str(child.pid)+'\\n').encode('ascii'))\n"
            " os._exit(0)\n"
        )
        parent = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=True,
        )
        child_pid = int(parent.stdout.strip())
        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, child_pid)
        if not handle:
            return
        try:
            status = ctypes.windll.kernel32.WaitForSingleObject(handle, 3000)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
        self.assertEqual(status, 0, "parent hard-exit left its child process running")

    @unittest.skipUnless(os.name == "nt", "Windows Job Object behavior")
    def test_job_assignment_failure_is_fail_closed(self) -> None:
        original = process_control._assign_windows_kill_job

        def fail_assignment(_process):
            raise ManagedProcessError("simulated job failure")

        process_control._assign_windows_kill_job = fail_assignment
        try:
            with self.assertRaises(ManagedProcessError):
                with managed_popen(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ):
                    pass
        finally:
            process_control._assign_windows_kill_job = original


if __name__ == "__main__":
    unittest.main()
