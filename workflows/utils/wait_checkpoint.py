"""Wait for a checkpoint (EMA or main) to appear on disk.

CLI tool for shell scripts that need to block until a training run produces
a specific checkpoint. Polls at a configurable interval and prints the resolved
path to stdout on success.

Usage:
    # Wait for EMA checkpoint (default)
    python -m workflows.utils.wait_checkpoint \
        --training-dir /path/to/train_run \
        --iteration 300000 \
        --interval 30 --timeout 3600

    # Wait for main (non-EMA) checkpoint
    python -m workflows.utils.wait_checkpoint \
        --training-dir /path/to/train_run \
        --iteration 300000 \
        --type ckpt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.workflow_utils.common import (
    CKPT_SUBDIR,
    EMA_SUBDIR,
    build_ckpt_path,
    build_ema_glob_pattern,
    find_ckpt,
    find_ema_ckpt,
)


def wait_for_checkpoint(
    training_dir: str,
    iteration: int,
    *,
    ckpt_type: str = "ema",
    ema_std: float = 0.050,
    interval: float = 60.0,
    timeout: float = 0.0,
    quiet: bool = False,
) -> str | None:
    """Poll until the target checkpoint appears.

    Args:
        training_dir: Root training output directory.
        iteration: Target optimizer step.
        ckpt_type: "ema" or "ckpt" (main checkpoint).
        ema_std: EMA standard deviation (only used when ckpt_type="ema").
        interval: Seconds between poll attempts.
        timeout: Maximum wait in seconds. 0 means wait indefinitely.
        quiet: If True, suppress progress messages.

    Returns:
        Resolved checkpoint path on success, None on timeout.
    """
    if ckpt_type == "ema":
        ema_dir = Path(training_dir) / EMA_SUBDIR
        description = build_ema_glob_pattern(ema_dir, iteration, ema_std)
        find_fn = lambda: find_ema_ckpt(training_dir, iteration, ema_std)
    elif ckpt_type == "ckpt":
        expected = build_ckpt_path(Path(training_dir) / CKPT_SUBDIR, iteration)
        description = str(expected)
        find_fn = lambda: find_ckpt(training_dir, iteration)
    else:
        raise ValueError(f"Unknown ckpt_type: {ckpt_type!r}. Must be 'ema' or 'ckpt'.")

    if not quiet:
        print(f"[wait_checkpoint] Waiting for: {description}", file=sys.stderr)
        if timeout > 0:
            print(f"[wait_checkpoint] Timeout: {timeout}s", file=sys.stderr)

    start = time.time()
    while True:
        result = find_fn()
        if result is not None:
            if not quiet:
                elapsed = time.time() - start
                print(
                    f"[wait_checkpoint] Found after {elapsed:.0f}s: {result}",
                    file=sys.stderr,
                )
            return result

        if timeout > 0 and (time.time() - start) >= timeout:
            if not quiet:
                print("[wait_checkpoint] Timeout reached.", file=sys.stderr)
            return None

        time.sleep(interval)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Wait for a checkpoint (EMA or main) to appear on disk.",
    )
    parser.add_argument(
        "--training-dir", required=True,
        help="Root training output directory (contains ema/ and ckpt/ subdirs).",
    )
    parser.add_argument(
        "--iteration", type=int, required=True,
        help="Target optimizer step (e.g. 300000).",
    )
    parser.add_argument(
        "--type", dest="ckpt_type", choices=["ema", "ckpt"], default="ema",
        help="Checkpoint type to wait for (default: ema).",
    )
    parser.add_argument(
        "--ema-std", type=float, default=0.050,
        help="EMA standard deviation; only used with --type=ema (default: 0.050).",
    )
    parser.add_argument(
        "--interval", type=float, default=60.0,
        help="Poll interval in seconds (default: 60).",
    )
    parser.add_argument(
        "--timeout", type=float, default=0.0,
        help="Max wait in seconds. 0 = wait forever (default: 0).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress messages; only print the path on success.",
    )

    args = parser.parse_args()

    result = wait_for_checkpoint(
        training_dir=args.training_dir,
        iteration=args.iteration,
        ckpt_type=args.ckpt_type,
        ema_std=args.ema_std,
        interval=args.interval,
        timeout=args.timeout,
        quiet=args.quiet,
    )

    if result is None:
        sys.exit(1)

    print(result)


if __name__ == "__main__":
    main()
