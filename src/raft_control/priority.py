from __future__ import annotations

import sys


def raise_process_priority() -> bool:
    """Put this process in the Windows HIGH priority class.

    Measured on the USB-6363 rig: under full CPU load a normal-priority process
    starved the DAQ stream even with a 500 ms buffer, while a high-priority one
    kept 100 ms of headroom.  Returns False where unsupported.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    HIGH_PRIORITY_CLASS = 0x80
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    return bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), HIGH_PRIORITY_CLASS))
