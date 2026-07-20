"""Mirror stdout/stderr to a timestamped log file via a `tee` subprocess.

The caller decides whether to invoke this (typically `if fabric.local_rank == 0`),
so this module does not probe rank environment variables itself. fd-level
redirection (`os.dup2` on fd 1/2) means C-level writes like NCCL_DEBUG /
NVTE_DEBUG are captured too, matching the behavior of a shell `... |& tee log`.

In addition to the pipe-based tee, two safeguards ensure that fatal error
information reaches the log file even when the process is killed before the
pipe buffer is drained:

1. ``sys.excepthook`` — writes the traceback directly to the log file for
   any unhandled Python exception (covers CUDA OOM, etc.).
2. ``faulthandler.enable`` — writes a minimal traceback directly to the log
   file on C-level fatal signals (SIGSEGV, SIGABRT, etc.).
"""

from __future__ import annotations

import atexit
import faulthandler
import os
import signal
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path


def setup_stdout_tee(log_dir: str | os.PathLike, *, node_rank: int = 0) -> Path:
    """Mirror stdout/stderr to `{log_dir}/run_<timestamp>_node{node_rank}.log`.

    Returns the log file path.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}_node{node_rank}.log"

    # --- pipe-based tee (normal path) ---
    # Let tee ignore SIGINT/SIGTERM/SIGHUP so it stays alive while Python
    # writes error tracebacks.  SIG_IGN is inherited across exec (POSIX).
    # tee exits naturally on EOF when the Python process closes the pipe.
    def _tee_ignore_signals():
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    tee = subprocess.Popen(
        ["tee", str(log_path)],
        stdin=subprocess.PIPE,
        preexec_fn=_tee_ignore_signals,
    )

    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(tee.stdin.fileno(), 1)
    os.dup2(tee.stdin.fileno(), 2)
    # Drop our extra pipe handle so tee sees EOF when this process exits.
    tee.stdin.close()

    # --- safeguard: direct-to-file for fatal errors ---
    _install_crash_safeguards(log_path)

    print(f"[stdout-tee] node{node_rank} mirror -> {log_path}")
    return log_path


def _install_crash_safeguards(log_path: Path) -> None:
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    log_file = os.fdopen(log_fd, "w", closefd=False)

    # 1. Python-level unhandled exceptions (CUDA OOM, etc.)
    _prev_excepthook = sys.excepthook

    def _excepthook(exc_type, exc_value, exc_tb):
        try:
            log_file.write("\n--- unhandled exception (direct-to-file) ---\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=log_file)
            log_file.flush()
            os.fsync(log_fd)
        except Exception:
            pass
        _prev_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    # 2. C-level fatal signals (SIGSEGV, SIGABRT, etc.)
    faulthandler.enable(file=log_fd)

    # 3. Flush on normal exit.
    def _flush_on_exit():
        try:
            log_file.flush()
            os.fsync(log_fd)
        except Exception:
            pass

    atexit.register(_flush_on_exit)


__all__ = ["setup_stdout_tee"]
