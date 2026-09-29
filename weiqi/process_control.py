"""Subprocess lifecycle helpers that prevent orphaned training process trees."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from collections.abc import Iterator
from typing import Any, Optional


class ManagedProcessError(RuntimeError):
    pass


class _WindowsKillJob:
    def __init__(self, handle: int) -> None:
        self.handle = handle

    def close(self) -> None:
        if not self.handle:
            return
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        kernel32.CloseHandle(ctypes.c_void_p(self.handle))
        self.handle = 0


def _assign_windows_kill_job(process: subprocess.Popen[Any]) -> Optional[_WindowsKillJob]:
    """Put a child in a kill-on-close Job Object when Windows permits it."""

    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        if process.poll() is not None:
            return None
        raise ManagedProcessError("无法创建 Windows 子进程保护 Job Object")
    information = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    information.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        handle, 9, ctypes.byref(information), ctypes.sizeof(information)
    ):
        kernel32.CloseHandle(handle)
        if process.poll() is not None:
            return None
        raise ManagedProcessError("无法配置 Windows 子进程保护 Job Object")
    if not kernel32.AssignProcessToJobObject(handle, wintypes.HANDLE(process._handle)):
        kernel32.CloseHandle(handle)
        if process.poll() is not None:
            return None
        raise ManagedProcessError("无法将外部训练进程加入 Windows Job Object")
    return _WindowsKillJob(int(handle))


def _terminate_process_tree(
    process: subprocess.Popen[Any], job: Optional[_WindowsKillJob]
) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        if job is not None:
            job.close()
        else:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                subprocess.run(
                    ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "nt":
        with contextlib.suppress(OSError):
            process.kill()
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)


def _graceful_console_interrupt(process: subprocess.Popen[Any]) -> bool:
    """Allow a shared-console Ctrl+C handler to finish before forced cleanup."""

    if process.poll() is not None:
        return True
    if os.name != "nt":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=30)
        return True
    except subprocess.TimeoutExpired:
        return False


@contextlib.contextmanager
def managed_popen(*args: Any, **kwargs: Any) -> Iterator[subprocess.Popen[Any]]:
    """Start a process whose entire child tree dies if this owner exits."""

    options = dict(kwargs)
    if os.name != "nt":
        options.setdefault("start_new_session", True)
    process = subprocess.Popen(*args, **options)
    try:
        try:
            job = _assign_windows_kill_job(process)
        except BaseException:
            _terminate_process_tree(process, None)
            raise
        try:
            yield process
        except KeyboardInterrupt:
            if not _graceful_console_interrupt(process):
                _terminate_process_tree(process, job)
            raise
        except BaseException:
            _terminate_process_tree(process, job)
            raise
        else:
            if process.poll() is None:
                _terminate_process_tree(process, job)
    finally:
        if "job" in locals() and job is not None:
            # Also removes any descendant that outlived an otherwise clean
            # immediate child process.
            job.close()


__all__ = ["ManagedProcessError", "managed_popen"]
