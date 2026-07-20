"""SwanLab logger helpers for workflows.

Ported from the legacy train pipeline (`archive/legacy/lit_gpt/utils.py::get_swanlab_id`
and `archive/legacy/pretrain/train_transmdm_vcc_dense_patch.py`). The logger is
plugged into Lightning Fabric via the standard `loggers=[...]` interface so the
train loop can call `fabric.log_dict(...)` without knowing about SwanLab.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
except ImportError:
    SwanLabLogger = None  # type: ignore[assignment]


_RUN_ID_PATCHED = False


def _ensure_unique_offline_run_id() -> None:
    """Make swanlab's offline/local run-id generation independent of the global RNG seed.

    swanlab's ``swanlab.data.namer.generate_run_id`` draws from the global ``random``
    module (``random.choices("a-z0-9", k=21)``). ``fabric.seed_everything(cfg.seed)``
    seeds that module, and in offline/local mode the run id comes from this function
    (cloud mode instead overwrites it with the server-assigned cuid in
    ``Mounter.execute``). So two runs sharing ``cfg.seed`` get the SAME run id — and
    offline mode ignores any explicitly passed id. Patch the generator to draw from OS
    entropy (``secrets``) so each run gets a unique id, without touching torch/numpy
    reproducibility. Idempotent and best-effort (never blocks training).
    """
    global _RUN_ID_PATCHED
    if _RUN_ID_PATCHED:
        return
    try:
        import secrets

        from swanlab.data import namer as _namer

        _alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        _namer.generate_run_id = lambda: "".join(
            secrets.choice(_alphabet) for _ in range(21)
        )
        _RUN_ID_PATCHED = True
    except Exception as exc:  # noqa: BLE001 - patch is best-effort, never block training
        print(f"[swanlab] could not patch generate_run_id for unique offline ids: {exc}")


def get_swanlab_id(
    swanlab_dir: str | os.PathLike,
    resume: bool,
    *,
    project: str,
    name: str,
) -> Optional[str]:
    """Resolve a SwanLab experiment id for resume, matching legacy semantics.

    Priority on resume:
      1. read id from `{swanlab_dir}/swanlab_info.json` if present;
      2. otherwise query SwanLab OpenApi for an experiment with the same name
         in the project, then persist its cuid to `swanlab_info.json`.
    """
    import swanlab

    swanlab_dir = str(swanlab_dir)
    info_path = os.path.join(swanlab_dir, "swanlab_info.json")
    swanlab_id: Optional[str] = None

    if not resume:
        return None

    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
            swanlab_id = info.get("swanlab_id")
        print(f"resume from swanlab_id {swanlab_id}")
        return swanlab_id

    # The lookup below uses swanlab's cloud OpenApi, which always logs in to
    # api.swanlab.cn and ignores SWANLAB_MODE. Skip it entirely outside cloud
    # mode, and degrade gracefully on any failure (e.g. the cloud being
    # unreachable) so a network hiccup never kills training.
    mode = os.environ.get("SWANLAB_MODE", "cloud").strip().lower()
    if mode not in ("", "cloud", "online"):
        return None

    target_experiment = None
    try:
        api = swanlab.OpenApi()
        experiments = api.list_experiments(project=project).data
        for exp in experiments:
            if exp.name == name:
                target_experiment = exp
                break
    except Exception as exc:  # noqa: BLE001 - resume-id lookup is best-effort
        print(
            f"[swanlab] cloud experiment lookup failed "
            f"({type(exc).__name__}: {exc}); proceeding without a resume id."
        )
        return None

    if target_experiment is not None:
        swanlab_id = target_experiment.cuid
        os.makedirs(swanlab_dir, exist_ok=True)
        with open(info_path, "w") as f:
            json.dump({"swanlab_id": swanlab_id}, f)
        print(f"save swanlab_id {swanlab_id}")

    return swanlab_id


def build_swanlab_logger(
    *,
    run_dir: str | os.PathLike,
    project: str,
    experiment_name: str,
    resume: bool,
):
    """Construct a SwanLabLogger rooted at `{run_dir}/swanlab`.

    Returns `None` when swanlab is not installed so the caller can drop it
    from the Fabric loggers list.
    """
    if SwanLabLogger is None:
        print("swanlab is not installed; skipping SwanLabLogger.")
        return None

    # Offline/local run ids come from swanlab's seed-dependent generate_run_id;
    # make them unique even under a fixed global seed (see helper docstring).
    _ensure_unique_offline_run_id()

    # swanlab only permits resume in cloud mode (swanlab.data.sdk.init raises
    # "You can only use resume in cloud mode." otherwise), and the cloud
    # resume-id lookup needs cloud access. In offline/local/disabled mode, log
    # locally without resuming so the run still starts on isolated (no-network) nodes.
    mode = os.environ.get("SWANLAB_MODE", "cloud").strip().lower()
    cloud = mode in ("", "cloud", "online")
    resume_cloud = bool(resume) and cloud

    swanlab_dir = str(Path(run_dir) / "swanlab")
    os.makedirs(swanlab_dir, exist_ok=True)
    swanlab_id = get_swanlab_id(
        swanlab_dir,
        resume_cloud,
        project=project,
        name=experiment_name,
    )
    return SwanLabLogger(
        experiment_name=experiment_name,
        logdir=swanlab_dir,
        project=project,
        id=swanlab_id,
        resume="allow" if resume_cloud else None,
    )


__all__ = ["build_swanlab_logger", "get_swanlab_id"]
