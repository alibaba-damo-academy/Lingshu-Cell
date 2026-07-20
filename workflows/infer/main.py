"""Inference workflow for reverse masked-diffusion single-cell generation.

Loads a trained checkpoint, iteratively unmasks tokens to generate
single-cell gene expression sequences, and saves results as h5ad.

Usage::

    # condition_csv accepts a file path ...
    python -m workflows.infer.main \\
        checkpoint.training_dir=/path/to/train_run \\
        task.condition_csv=/path/to/conditions.csv

    # ... or inline CSV content (use Hydra single quotes to escape commas)
    python -m workflows.infer.main \\
        checkpoint.training_dir=/path/to/train_run \\
        "task.condition_csv='tissue,n_cells
    heart,1000
    PBMC,2000'"
"""

from __future__ import annotations

import glob
import math
import os
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

# Suppress anndata noise: index-type coercion and non-unique obs_names during batch concat/merge.
warnings.filterwarnings("ignore", message=".*Transforming to str index.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*names are not unique.*", category=UserWarning)

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
import numpy as np
import pandas as pd
import anndata as ad

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modules.common import TokenSequence
from src.modules.common.interfaces import ModelInput
from src.modules.samplers.clue_injection import (
    apply_clue_prefill,
    extract_clues,
    load_gene_id_name_mapping,
)
from src.modules.samplers.default import diff_sample
from src.workflow_utils import (
    EMA_SUBDIR,
    build_ema_glob_pattern,
    build_model_sequence_format,
    find_ema_ckpt,
    infer_patch_aligned_block_size,
    instantiate_model,
    resolve_mask_token_id,
    setup_stdout_tee,
    validated_main,
)
from workflows.infer.config_schema import InferConfig

CFG_TYPE_TO_UNCOND_WORD: Dict[str, Optional[str]] = {
    "cfg_mask_condition": None,
    "cfg_non_targeting": "non-targeting",
    "cfg_PBS": "PBS",
    "cfg_unknown": "<unk_gene>",
}


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../configs", config_name="infer/default")
@validated_main(InferConfig)
def main(cfg: InferConfig) -> int:
    """Hydra entry point for the inference workflow."""
    torch.set_float32_matmul_precision("high")
    infer_workflow(cfg)
    return 0


# ---------------------------------------------------------------------------
# Workflow assembly
# ---------------------------------------------------------------------------


def infer_workflow(cfg: InferConfig) -> None:
    """Assemble components and run the generation loop.

    Assembly order:
    1. Resolve device/dtype.
    2. Optionally load tokenizer/model config from training directory.
    3. Build tokenizer.
    4. Build model.
    5. Load checkpoint.
    6. Parse condition CSV and generate tasks.
    7. Run generation loop with periodic saving.
    """
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)

    # Override tokenizer/model config from training directory (new format only).
    # Legacy checkpoints do not have a sibling .hydra config; the user supplies
    # tokenizer/model via a dedicated legacy config file.
    is_legacy = bool(cfg.checkpoint.is_legacy)
    training_dir = getattr(cfg.checkpoint, "training_dir", "")
    if not is_legacy and training_dir:
        print(f"Loading tokenizer/model config from training dir: {training_dir}")
        tok_cfg, model_cfg = load_training_config(training_dir)
        with open_dict(cfg):
            cfg.tokenizer = tok_cfg
            cfg.model = model_cfg

    # Legacy: auto-parse iteration/ema_std from the checkpoint filename so
    # downstream interpolation (e.g. sample_dir) sees meaningful values.
    # If iteration/ema_std are already set in the config (e.g. for renamed
    # checkpoints whose filenames no longer match the legacy pattern),
    # skip the auto-parse step.
    if is_legacy:
        if not cfg.checkpoint.model_path:
            raise ValueError(
                "checkpoint.is_legacy=true requires an explicit checkpoint.model_path."
            )
        from src.compat.legacy_checkpoints import identify_legacy_checkpoint
        try:
            meta = identify_legacy_checkpoint(cfg.checkpoint.model_path)
            with open_dict(cfg):
                cfg.checkpoint.iteration = meta.iteration
                cfg.checkpoint.ema_std = meta.ema_std
            print(
                f"Legacy ckpt metadata parsed: iteration={meta.iteration}, "
                f"ema_std={meta.ema_std}"
            )
        except ValueError:
            # Renamed checkpoint — use iteration/ema_std from config as-is.
            print(
                f"Legacy ckpt filename does not match legacy pattern; "
                f"using config values: iteration={cfg.checkpoint.iteration}, "
                f"ema_std={cfg.checkpoint.ema_std}"
            )

    # Tokenizer.
    tokenizer = instantiate(cfg.tokenizer)
    model_sequence_format = build_model_sequence_format(tokenizer)
    model_vocab_size = int(tokenizer.vocab_size)

    # Block size.
    raw_patch_size = OmegaConf.select(cfg, "model.config.patch_size", default=1)
    patch_size = int(raw_patch_size) if raw_patch_size is not None else 1
    block_size = infer_patch_aligned_block_size(
        seq_length=int(tokenizer.seq_length),
        condition_prefix_length=int(tokenizer.condition_prefix_length),
        patch_size=patch_size,
    )

    # Model.
    model = instantiate_model(
        cfg.model,
        sequence_format=model_sequence_format,
        vocab_size=model_vocab_size,
        block_size=block_size,
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Load checkpoint.
    model_path = resolve_model_path(cfg)
    print(f"Loading checkpoint from: {model_path}")
    load_model_checkpoint(model, model_path, device=device, is_legacy=is_legacy)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, device={device}, dtype={dtype}")
    print(f"Tokenizer: vocab_size={model_vocab_size}, seq_length={tokenizer.seq_length}")

    # Sampler config (no class instantiation — diff_sample is called directly).
    sampler_cfg = cfg.sampler
    mask_token_id = resolve_mask_token_id(model=model)

    # Build tasks from condition CSV (full, before subtask split).
    full_tasks_df = build_task_dataframe(cfg, tokenizer)
    run_dir = resolve_infer_run_dir(cfg)

    # Mirror stdout/stderr to a log file in run_dir.
    setup_stdout_tee(run_dir / "logging")

    # Apply subtask split for this worker.
    current_subtask, total_subtasks = parse_subtask(cfg.task.subtask)
    if total_subtasks > 1:
        tasks_df = _apply_subtask_split(full_tasks_df, current_subtask, total_subtasks)
        print(f"Subtask {current_subtask}/{total_subtasks}: {tasks_df['n_cells'].sum()} cells")
    else:
        tasks_df = full_tasks_df

    # Generation loop.
    run_generation_loop(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        sampler_cfg=sampler_cfg,
        tasks_df=tasks_df,
        full_tasks_df=full_tasks_df,
        total_subtasks=total_subtasks,
        run_dir=run_dir,
        block_size=block_size,
        mask_token_id=mask_token_id,
        device=device,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Device / dtype resolution
# ---------------------------------------------------------------------------


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to a torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    """Resolve dtype string to a torch.dtype."""
    if dtype_str == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_str]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_training_config(training_dir: str) -> tuple:
    """Load tokenizer and model config from a training directory.

    Reads the Hydra-saved ``config.yaml`` and ``overrides.yaml`` from the
    first rank directory found under ``{training_dir}/hydra/``.  Overrides
    prefixed with ``tokenizer.`` or ``model.`` are merged on top.

    Args:
        training_dir: Path to the training output directory.

    Returns:
        Tuple of ``(tokenizer_cfg, model_cfg)`` as DictConfig objects.
    """
    training_path = Path(training_dir)
    hydra_configs = sorted(glob.glob(
        str(training_path / "hydra" / "*" / ".hydra" / "config.yaml")
    ))
    if not hydra_configs:
        raise FileNotFoundError(
            f"No Hydra config found in {training_dir}/hydra/*/.hydra/config.yaml"
        )

    config_path = Path(hydra_configs[0])
    print(f"  Using training config: {config_path}")
    train_cfg = OmegaConf.load(str(config_path))

    # Apply tokenizer/model overrides from overrides.yaml.
    overrides_path = config_path.parent / "overrides.yaml"
    if overrides_path.exists():
        overrides_list = OmegaConf.load(str(overrides_path))
        filtered = [
            o for o in overrides_list
            if o.startswith("tokenizer.") or o.startswith("model.")
        ]
        if filtered:
            print(f"  Applying {len(filtered)} tokenizer/model overrides")
            override_cfg = OmegaConf.from_dotlist(filtered)
            train_cfg = OmegaConf.merge(train_cfg, override_cfg)

    return train_cfg.tokenizer, train_cfg.model


def resolve_model_path(cfg: InferConfig) -> str:
    """Resolve the model checkpoint path from config.

    Legacy mode (``checkpoint.is_legacy=true``): returns
    ``checkpoint.model_path`` directly. It must be set explicitly.

    New-format mode: uses ``training_dir`` + ``iteration`` + ``ema_std`` to
    glob for ``{training_dir}/ema/step-{iteration:06d}-ema-*samples-{std}std.pth``.
    An explicit ``model_path`` takes priority if set.
    """
    if cfg.checkpoint.is_legacy:
        if not cfg.checkpoint.model_path:
            raise ValueError(
                "checkpoint.is_legacy=true requires an explicit checkpoint.model_path."
            )
        if not Path(cfg.checkpoint.model_path).is_file():
            raise FileNotFoundError(
                f"Legacy checkpoint not found: {cfg.checkpoint.model_path}"
            )
        return cfg.checkpoint.model_path

    if cfg.checkpoint.model_path:
        return cfg.checkpoint.model_path

    ema_std = cfg.checkpoint.ema_std
    iteration = cfg.checkpoint.iteration

    training_dir = getattr(cfg.checkpoint, "training_dir", "")
    if not training_dir:
        raise ValueError(
            "checkpoint.training_dir is required for non-legacy inference. "
            "Set it via CLI: checkpoint.training_dir=/path/to/train_run"
        )

    result = find_ema_ckpt(training_dir, iteration, ema_std)
    if result is None:
        pattern = build_ema_glob_pattern(
            Path(training_dir) / EMA_SUBDIR, iteration, ema_std,
        )
        raise FileNotFoundError(
            f"No EMA checkpoint found matching: {pattern}\n"
            f"Set checkpoint.model_path explicitly or check iteration/ema_std."
        )
    return result


def load_model_checkpoint(
    model: torch.nn.Module,
    model_path: str,
    *,
    device: torch.device,
    is_legacy: bool = False,
) -> None:
    """Load model weights from a checkpoint file.

    Supports ``.pth`` (torch.save) and ``.safetensors`` formats.
    Handles both raw state dicts and ``{"model": state_dict}`` wrappers.

    Args:
        model: Target module to populate.
        model_path: Checkpoint file path.
        device: Target device for the state dict.
        is_legacy: If True, convert legacy (old repo) key naming before
            loading. The caller must set this explicitly based on the
            config; no auto-detection is performed.
    """
    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(model_path, device=str(device))
    else:
        raw = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = raw.get("model", raw) if isinstance(raw, dict) else raw
        if is_legacy:
            from src.compat.legacy_checkpoints import convert_legacy_state_dict
            print("  Converting legacy checkpoint keys...")
            state_dict = convert_legacy_state_dict(state_dict)

    # Allow missing TE _extra_state keys (zero-size, non-essential).
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    te_missing = [k for k in missing if "te_dot_product_attn._extra_state" in k]
    real_missing = [k for k in missing if k not in te_missing]
    if real_missing:
        raise RuntimeError(
            f"Missing keys when loading checkpoint: {real_missing}"
        )
    if unexpected:
        # Filter TE extra state from legacy that new model doesn't have.
        real_unexpected = [
            k for k in unexpected
            if "te_dot_product_attn._extra_state" not in k
        ]
        if real_unexpected:
            raise RuntimeError(
                f"Unexpected keys when loading checkpoint: {real_unexpected}"
            )
    if te_missing:
        print(f"  Skipped {len(te_missing)} TE _extra_state keys (non-essential).")


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------


def resolve_infer_run_dir(cfg: InferConfig) -> Path:
    """Resolve the inference output directory.

    Returns ``Path(savedir) / sample_dir``.  ``sample_dir`` is always
    non-empty: the YAML default uses Hydra interpolation to auto-generate
    a name from checkpoint/sampler parameters, and CLI users can override
    it with an explicit value.

    Args:
        cfg: Resolved inference config.

    Returns:
        Absolute path to the run output directory.
    """
    return Path(cfg.output.workdir) / cfg.output.sample_dir



# ---------------------------------------------------------------------------
# Task building from condition CSV
# ---------------------------------------------------------------------------


def build_task_dataframe(
    cfg: InferConfig,
    tokenizer: object,
) -> pd.DataFrame:
    """Build a task DataFrame from condition CSV.

    ``task.condition_csv`` is required and accepts either a file path or
    inline CSV content. The CSV must contain an ``n_cells`` column and all
    condition columns declared by the tokenizer's ``required_obs_keys``.

    Args:
        cfg: Inference config.
        tokenizer: Tokenizer with ``required_obs_keys`` attribute.

    Returns:
        DataFrame with columns from CSV plus ``seed_start``.
    """
    from io import StringIO

    condition_csv = cfg.task.condition_csv
    if not condition_csv or not condition_csv.strip():
        raise ValueError(
            "task.condition_csv is required. Provide a file path or inline CSV content."
        )

    if os.path.isfile(condition_csv):
        print(f"Reading conditions from file: {condition_csv}")
        tasks_df = pd.read_csv(condition_csv)
    else:
        print("Reading conditions from inline CSV content")
        tasks_df = pd.read_csv(StringIO(condition_csv))

    # Validate required columns.
    if "n_cells" not in tasks_df.columns:
        raise ValueError("Condition CSV must contain 'n_cells' column.")

    required_obs = getattr(tokenizer, "required_obs_keys", ())
    for key in required_obs:
        if key not in tasks_df.columns:
            raise ValueError(
                f"Condition CSV missing required column '{key}' "
                f"(from tokenizer.required_obs_keys)."
            )

    # Pre-assign seeds.
    base_seed = cfg.task.base_seed
    cumulative_seed = base_seed
    for i in range(len(tasks_df)):
        tasks_df.at[i, "seed_start"] = int(cumulative_seed)
        cumulative_seed += int(tasks_df.at[i, "n_cells"])

    total = tasks_df["n_cells"].sum()
    print(f"Will generate {total} samples across {len(tasks_df)} conditions.")
    return tasks_df


def parse_subtask(subtask_str: str) -> Tuple[int, int]:
    """Parse ``"3of16"`` into ``(3, 16)``."""
    if not subtask_str or subtask_str == "1of1":
        return 1, 1
    parts = subtask_str.split("of")
    if len(parts) != 2:
        raise ValueError(f"Subtask must be 'currentOFtotal', got: {subtask_str!r}")
    current, total = int(parts[0]), int(parts[1])
    if not (1 <= current <= total):
        raise ValueError(f"Invalid subtask range: {current}/{total}")
    return current, total


def split_task_cells(
    task_row: pd.Series,
    split_index: int,
    *,
    num_splits: Optional[int] = None,
    split_size: Optional[int] = None,
) -> Optional[pd.Series]:
    """Split a task's cells for subtask or periodic-save division.

    Args:
        task_row: DataFrame row with ``n_cells`` and ``seed_start``.
        split_index: 0-based split index.
        num_splits: Total splits (for worker distribution).
        split_size: Fixed split size (for ``save_every_n_samples``).

    Returns:
        Split row, or ``None`` if this split has no cells.
    """
    if (num_splits is None) == (split_size is None):
        raise ValueError("Exactly one of num_splits or split_size must be specified.")

    n = int(task_row["n_cells"])
    seed_start = int(task_row["seed_start"])

    if n <= 0:
        return None

    if num_splits is not None:
        offsets = np.array_split(np.arange(n), num_splits)[split_index]
        if len(offsets) == 0:
            return None
        offset_start = int(offsets[0])
        split_n = len(offsets)
    else:
        offset_start = split_index * split_size
        split_n = min(split_size, n - offset_start)
        if split_n <= 0:
            return None

    row = task_row.copy()
    row["n_cells"] = split_n
    row["seed_start"] = seed_start + offset_start
    return row


def _apply_subtask_split(
    tasks_df: pd.DataFrame,
    current: int,
    total: int,
) -> pd.DataFrame:
    """Filter tasks for the current subtask."""
    rows = []
    for _, row in tasks_df.iterrows():
        split = split_task_cells(row, current - 1, num_splits=total)
        if split is not None:
            rows.append(split)
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CFG unconditional generator builders
# ---------------------------------------------------------------------------


def build_cfg_uncond_input_generator(
    cfg_token_type: str,
    *,
    tokenizer: object,
    prefix_tokens: List[int],
    task_row: pd.Series,
    mask_token_id: int,
    device: torch.device,
    perturbation_metadata: Optional[Dict[str, torch.Tensor]] = None,
    model: Any = None,
    cfg_uncond_overrides: Optional[Dict[str, str]] = None,
) -> Optional[Callable[[ModelInput], ModelInput]]:
    """Build a CFG unconditional-input generator operating on ``ModelInput``.

    The returned callable transforms a conditional ``ModelInput`` into an
    unconditional one suitable for the second forward pass in
    classifier-free guidance. Both the token prefix (in
    ``noisy_token_ids``) and any conditioning metadata (e.g.
    ``drug_indices``, ``dose_values``) may be modified depending on
    ``cfg_token_type``; non-condition metadata like ``mask_indices`` is
    always preserved.

    Args:
        cfg_token_type: One of:

            - ``cfg_mask_condition``: replace internal condition tokens with
              ``mask_token_id`` (works for any condition format). Metadata
              is not modified.
            - ``cfg_general_control``: replace the ConditionFormat's
              ControlIndicator slot with its declared ``control_value`` while
              keeping all other slots' values from ``task_row`` (token-level).
              Additionally, if perturbation metadata is present, replaces
              ``drug_indices`` with the control drug index (index 0 = DMSO_TF
              by Tahoe convention) and ``dose_values`` with the normalized
              zero dose. Generalises across datasets — no per-dataset
              hardcoded keys; only requires
              ``tokenizer.condition_format.control_indicator`` to be set.
            - ``cfg_non_targeting`` / ``cfg_PBS`` / ``cfg_unknown``: VCC-style
              fixed uncond word; replaces ``target_gene`` with the mapped word
              and keeps ``cell_type``. Hardcodes VCC condition keys; for new
              datasets prefer ``cfg_general_control``. Metadata is not
              modified.

        tokenizer: Tokenizer for encoding alternate conditions.
        prefix_tokens: Original condition prefix token ids.
        task_row: Current task row with condition columns.
        mask_token_id: Mask token id.
        device: Torch device.
        perturbation_metadata: Per-task metadata dict (``drug_indices``,
            ``dose_values``) from ``_build_perturbation_metadata``, or
            ``None`` for non-perturbation models. Used by
            ``cfg_general_control`` to build control metadata.
        model: The loaded model. Used by ``cfg_general_control`` to
            compute the normalized zero dose via
            ``model.perturbation_embedder``.
        cfg_uncond_overrides: A dict ``{condition_obs_key: uncond_value}``
            applied in the ``cfg_general_control`` branch to set those
            condition slots to their unconditional value on top of the
            existing ``control_indicator`` neutralization.  General-purpose:
            works for any condition slot (not just genes).  Keys are
            validated against the format's slot ``obs_key`` values so a
            typo hard-errors instead of silently being a no-op.  Empty dict
            or ``None`` preserves byte-identical behaviour to the existing
            code path (back-compat).

    Returns:
        Callable ``(ModelInput) -> ModelInput`` or ``None`` if CFG is
        disabled / the cfg_token_type is unrecognised.

    Raises:
        ValueError: If ``cfg_general_control`` is selected but the tokenizer
            does not expose a ``ConditionFormat`` with a ``ControlIndicator``.
    """
    # ---- cfg_mask_condition: mask prefix tokens, metadata untouched ----
    if cfg_token_type == "cfg_mask_condition":
        prefix_len = len(prefix_tokens)

        def _mask_condition(cond_input: ModelInput) -> ModelInput:
            un_x = cond_input.noisy_token_ids.clone()
            un_x[:, 1 : prefix_len - 1] = mask_token_id
            return ModelInput(noisy_token_ids=un_x, metadata=dict(cond_input.metadata))

        return _mask_condition

    # ---- cfg_general_control: token prefix + metadata replacement ----
    if cfg_token_type == "cfg_general_control":
        cf = getattr(tokenizer, "condition_format", None)
        if cf is None:
            raise ValueError(
                "cfg_general_control requires the tokenizer to expose a "
                "`condition_format` attribute; current tokenizer has none."
            )
        ci = getattr(cf, "control_indicator", None)
        if ci is None:
            raise ValueError(
                "cfg_general_control requires "
                "`tokenizer.condition_format.control_indicator` to be set; "
                "current ConditionFormat has none."
            )

        # Pre-compute the unconditional token prefix (frozen per-task).
        uncond_conditions = cf.resolve_obs_values(task_row)
        uncond_conditions[ci.obs_key] = ci.control_value

        # Apply declarative per-obs-key unconditional overrides (general
        # multi-condition CFG): neutralize each listed condition slot to its
        # given value, on top of the control_indicator. Validate keys against the
        # format's slot obs_keys so a typo hard-errors instead of silently no-op.
        if cfg_uncond_overrides:
            valid_keys = {s.obs_key for s in cf.slots if s.obs_key}
            unknown = [k for k in cfg_uncond_overrides if k not in valid_keys]
            if unknown:
                raise ValueError(
                    f"cfg_uncond_overrides has unknown condition key(s) {unknown}; "
                    f"valid keys (condition obs_keys): {sorted(valid_keys)}"
                )
            for k, v in cfg_uncond_overrides.items():
                uncond_conditions[k] = str(v)

        uncond_prefix_tokens = tokenizer.encode_condition(uncond_conditions)
        uncond_prefix_tensor = torch.tensor(
            uncond_prefix_tokens, dtype=torch.long, device=device
        )

        # Pre-compute control metadata replacements (frozen per-task).
        control_drug_index: Optional[int] = None
        control_dose_value: Optional[float] = None
        if perturbation_metadata is not None:
            drug_index_map = getattr(tokenizer, "drug_index_map", {})
            control_drug_index = drug_index_map.get(ci.control_value, 0)

            if model is not None and hasattr(model, "perturbation_embedder"):
                from src.modules.datasets.utils_tahoe_drug import (
                    compute_normalized_dose,
                )
                emb = model.perturbation_embedder
                control_dose_value = compute_normalized_dose(
                    0.0,
                    emb.dose_log1p_mean.item(),
                    emb.dose_log1p_std.item(),
                )

        def _general_control(cond_input: ModelInput) -> ModelInput:
            # 1) Replace token prefix.
            un_x = cond_input.noisy_token_ids.clone()
            un_x[:, : len(uncond_prefix_tokens)] = uncond_prefix_tensor

            # 2) Replace perturbation metadata while preserving the rest.
            un_meta = dict(cond_input.metadata)
            if control_drug_index is not None and "drug_indices" in un_meta:
                un_meta["drug_indices"] = torch.full_like(
                    un_meta["drug_indices"], control_drug_index,
                )
            if control_dose_value is not None and "dose_values" in un_meta:
                un_meta["dose_values"] = torch.full_like(
                    un_meta["dose_values"], control_dose_value,
                )

            return ModelInput(noisy_token_ids=un_x, metadata=un_meta)

        return _general_control

    # ---- VCC-style fixed uncond word (legacy) ----
    uncond_word = CFG_TYPE_TO_UNCOND_WORD.get(cfg_token_type)
    if uncond_word is None:
        return None

    target_key = getattr(tokenizer, "condition_target_obs_key", "target_gene")
    cell_type_key = getattr(tokenizer, "condition_cell_type_obs_key", "cell_type")
    uncond_conditions = {
        target_key: uncond_word,
        cell_type_key: str(task_row[cell_type_key]),
    }
    uncond_prefix_tokens = tokenizer.encode_condition(uncond_conditions)
    uncond_prefix_tensor = torch.tensor(
        uncond_prefix_tokens, dtype=torch.long, device=device
    )

    def _replace_prefix(cond_input: ModelInput) -> ModelInput:
        un_x = cond_input.noisy_token_ids.clone()
        un_x[:, : len(uncond_prefix_tokens)] = uncond_prefix_tensor
        return ModelInput(noisy_token_ids=un_x, metadata=dict(cond_input.metadata))

    return _replace_prefix


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _build_perturbation_metadata(
    *,
    model: Any,
    tokenizer: Any,
    task_row: "pd.Series",
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build drug/dose metadata tensors for perturbation models.

    Returns ``None`` for non-perturbation models (which is the case for
    all three paper experiments: PARSE PBMC, VCC, PARSE cytokine).

    Args:
        model: The loaded model (checked for ``perturbation_embedder``).
        tokenizer: Tokenizer instance.
        task_row: One row from the condition DataFrame.
        device: Target device for tensors.

    Returns:
        ``None`` (drug perturbation models are not supported in this release).
    """
    return None


def _build_initial_sequence(
    *,
    prefix_tokens: List[int],
    batch_size: int,
    block_size: int,
    mask_token_id: int,
    num_init_padding: int,
    eos_id: int,
    condition_prefix_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the initial ``(B, L)`` tensor with condition prefix and mask fill.

    Args:
        prefix_tokens: Encoded condition prefix token ids.
        batch_size: Number of samples in this batch.
        block_size: Full padded sequence length.
        mask_token_id: Token id for masked positions.
        num_init_padding: Number of trailing EOS padding positions.
        eos_id: End-of-sequence token id.
        condition_prefix_length: Length of the condition prefix.
        device: Torch device.

    Returns:
        ``(B, block_size)`` tensor ready for ``diff_sample``.
    """
    x0 = torch.full(
        (batch_size, block_size), mask_token_id,
        dtype=torch.long, device=device,
    )
    prefix_ids = torch.tensor(
        prefix_tokens[:condition_prefix_length],
        dtype=torch.long, device=device,
    )
    x0[:, : len(prefix_ids)] = prefix_ids.unsqueeze(0)
    if num_init_padding > 0:
        x0[:, -num_init_padding:] = eos_id
    return x0


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------


def run_generation_loop(
    *,
    cfg: InferConfig,
    model: torch.nn.Module,
    tokenizer: object,
    sampler_cfg: Any,
    tasks_df: pd.DataFrame,
    full_tasks_df: pd.DataFrame,
    total_subtasks: int,
    run_dir: Path,
    block_size: int,
    mask_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Run the main generation loop over all tasks.

    For each task row:
    1. Encode condition prefix.
    2. Optionally split into sub-batches for periodic saving.
    3. Run inner batch loop calling ``diff_sample`` directly.
    4. Decode and save as h5ad.

    After all tasks complete, checks whether all expected batch files
    (derived from ``full_tasks_df``) are present and merges them if so.

    Args:
        cfg: Inference config.
        model: Loaded model on the target device.
        tokenizer: Tokenizer for encoding/decoding.
        sampler_cfg: Sampler config namespace (read-only).
        tasks_df: Task DataFrame for this subtask (after split).
        full_tasks_df: Full task DataFrame (before subtask split),
            used to determine merge completeness.
        total_subtasks: Total number of subtask workers (1 if no split).
        run_dir: Output directory root.
        block_size: Padded sequence length.
        mask_token_id: Mask token id.
        device: Torch device.
        dtype: Torch dtype for autocast.
    """
    cfg_scale = float(sampler_cfg.cfg_scale)
    cfg_token_type = str(sampler_cfg.cfg_token_type)

    # Read declarative per-obs-key unconditional overrides for CFG (general
    # multi-condition support). Empty dict = no-op, back-compat default.
    _uncond_ov = sampler_cfg.get("cfg_uncond_overrides", {})
    cfg_uncond_overrides: Dict[str, str] = (
        {str(k): str(v) for k, v in OmegaConf.to_container(_uncond_ov, resolve=True).items()}
        if _uncond_ov else {}
    )

    condition_prefix_length = int(tokenizer.condition_prefix_length)
    raw_padding = sampler_cfg.num_init_padding
    if raw_padding is None:
        num_init_padding = block_size - int(tokenizer.seq_length)
    else:
        num_init_padding = int(raw_padding)
    if num_init_padding != block_size - int(tokenizer.seq_length):
        print(
            f"WARNING: num_init_padding={num_init_padding} but "
            f"block_size - seq_length={block_size - int(tokenizer.seq_length)}. "
            f"Misalignment with training padding."
        )
    eos_id = int(tokenizer.eos_id)

    # --- Clue-based biology prior injection setup ---
    clue_csv_path = str(sampler_cfg.get("clue_csv", "") or "")
    clue_target_obs_key = str(sampler_cfg.get("clue_target_obs_key", "target_gene"))
    condition_target_count = int(sampler_cfg.get("condition_target_count", -1))
    clue_overwrite_target = bool(sampler_cfg.get("clue_overwrite_target", True))
    clues_dict: Dict[str, list] = {}

    if clue_csv_path:
        # Load gene ID mapping
        clue_gene_id_csv = str(sampler_cfg.get("clue_gene_id_name_csv", ""))
        if not clue_gene_id_csv:
            raise ValueError(
                "sampler.clue_gene_id_name_csv is required when clue_csv is set."
            )
        gene_ensembl_to_name = load_gene_id_name_mapping(clue_gene_id_csv)

        # Load clue CSV
        print(f"Loading clue CSV: {clue_csv_path}")
        clues_df = pd.read_csv(clue_csv_path)

        # Collect all target gene names from the task DataFrame
        all_target_genes = tasks_df[clue_target_obs_key].unique().tolist()

        # Parse filter rule sets and count rules from config
        _filter_rules_raw = sampler_cfg.get("clue_filter_rule_sets", [])
        filter_rule_sets: Optional[List[str]] = (
            [str(r) for r in OmegaConf.to_container(_filter_rules_raw, resolve=True)]
            if _filter_rules_raw else None
        )
        _count_rules_raw = sampler_cfg.get("clue_count_rules", ["count_1"])
        count_rules = [str(r) for r in OmegaConf.to_container(_count_rules_raw, resolve=True)]

        gathering_mode = str(sampler_cfg.get("clue_gathering_mode", "any"))
        group_col = str(sampler_cfg.get("clue_group_col", "cell_type"))
        use_ensembl = bool(sampler_cfg.get("clue_use_ensembl_as_target", True))

        clues_dict = extract_clues(
            target_genes=all_target_genes,
            clues_df=clues_df,
            gene_ensembl_to_name=gene_ensembl_to_name,
            filter_rule_sets=filter_rule_sets,
            count_rules=count_rules,
            gathering_mode=gathering_mode,
            group_col=group_col,
            use_ensembl_as_target=use_ensembl,
        )
        print(f"Clue injection enabled: {len(clues_dict)} targets with clues, "
              f"condition_target_count={condition_target_count}")
    else:
        print("Clue injection: disabled (no clue_csv configured)")

    batch_dir = run_dir / "batches_lower"
    batch_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(cfg.task.save_every_n_samples)
    batch_size = int(cfg.task.batch_size)

    # Optional knobs (default empty = legacy behaviour):
    #   filename_columns: short id columns used to name output files.
    #   obs_passthrough_columns: extra CSV columns copied verbatim into gen obs.
    filename_columns = [str(c) for c in (cfg.task.get("filename_columns", []) or [])]
    obs_passthrough_columns = [
        str(c) for c in (cfg.task.get("obs_passthrough_columns", []) or [])
    ]

    total_cells = int(tasks_df["n_cells"].sum())
    from tqdm import tqdm

    pbar = tqdm(total=total_cells, desc="Generating cells")
    t0 = time.perf_counter()

    for task_idx, task_row in tasks_df.iterrows():
        n_cells = int(task_row["n_cells"])
        seed_start = int(task_row["seed_start"])

        if n_cells <= 0:
            continue

        # Build conditions via condition_format.resolve_obs_values, which
        # auto-fills override_value / default_value / alias chain (matches
        # the dataset/training path).  This is backward-compatible: tokenizers
        # without override_value slots produce the same {obs_key: csv_value}
        # mapping the old dict-comprehension code produced.
        conditions = tokenizer.condition_format.resolve_obs_values(task_row)
        prefix_tokens = tokenizer.encode_condition(conditions)

        # Build perturbation metadata for drug/dose injection (if applicable).
        perturbation_metadata = _build_perturbation_metadata(
            model=model, tokenizer=tokenizer, task_row=task_row, device=device,
        )

        # Build CFG generator if needed.
        cfg_uncond_input_generator = None
        if cfg_scale > 0.0:
            cfg_uncond_input_generator = build_cfg_uncond_input_generator(
                cfg_token_type,
                tokenizer=tokenizer,
                prefix_tokens=prefix_tokens,
                task_row=task_row,
                mask_token_id=mask_token_id,
                device=device,
                perturbation_metadata=perturbation_metadata,
                model=model,
                cfg_uncond_overrides=cfg_uncond_overrides,
            )

        # Split into sub-batches for periodic saving.
        if save_every > 0 and n_cells > save_every:
            n_splits = math.ceil(n_cells / save_every)
        else:
            n_splits = 1

        for split_idx in range(n_splits):
            if n_splits > 1:
                split_row = split_task_cells(
                    task_row, split_idx, split_size=save_every
                )
                if split_row is None:
                    continue
                split_n = int(split_row["n_cells"])
                split_seed = int(split_row["seed_start"])
            else:
                split_n = n_cells
                split_seed = seed_start

            # Check for existing output (resume support).
            file_prefix = _make_file_prefix(task_row, filename_columns)
            file_path = batch_dir / f"{file_prefix}_seed-{split_seed:09d}-{split_seed + split_n:09d}.h5ad"
            if file_path.exists():
                tqdm.write(f"Skipping (exists): {file_path.name}")
                pbar.update(split_n)
                continue

            # Inner batch loop: call diff_sample directly.
            generated = 0
            all_output_ids: List[torch.Tensor] = []
            while generated < split_n:
                actual_bs = min(batch_size, split_n - generated)
                seed = split_seed + generated
                torch.manual_seed(seed)

                x0 = _build_initial_sequence(
                    prefix_tokens=prefix_tokens,
                    batch_size=actual_bs,
                    block_size=block_size,
                    mask_token_id=mask_token_id,
                    num_init_padding=num_init_padding,
                    eos_id=eos_id,
                    condition_prefix_length=condition_prefix_length,
                    device=device,
                )

                # --- Apply clue-based biology prior pre-fill ---
                if clues_dict:
                    task_target = str(task_row.get(clue_target_obs_key, ""))
                    if task_target:
                        n_placed = apply_clue_prefill(
                            x0=x0,
                            target_gene=task_target,
                            clues_dict=clues_dict,
                            tokenizer=tokenizer,
                            condition_prefix_length=condition_prefix_length,
                            condition_target_count=condition_target_count,
                            clue_overwrite_target=clue_overwrite_target,
                        )
                        if generated == 0 and pbar.n == 0:
                            tqdm.write(
                                f"Clue prefill: target='{task_target}', "
                                f"placed={n_placed} positions"
                            )

                # Build ModelInput: x0 + batch-expanded perturbation metadata.
                batch_metadata: dict = {}
                if perturbation_metadata is not None:
                    batch_metadata = {
                        k: v.expand(actual_bs) if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v
                        for k, v in perturbation_metadata.items()
                    }

                initial_input = ModelInput(
                    noisy_token_ids=x0,
                    metadata=batch_metadata,
                )

                output_ids = diff_sample(
                    model=model,
                    initial_input=initial_input,
                    mask_token_id=mask_token_id,
                    config=sampler_cfg,
                    tokenizer=tokenizer,
                    device=str(device),
                    device_type=device.type,
                    dtype=dtype,
                    cfg_uncond_input_generator=cfg_uncond_input_generator,
                    temperature=1.0,
                    print_info=(pbar.n == 0 and generated == 0),
                )

                all_output_ids.append(output_ids.cpu())
                generated += actual_bs
                pbar.update(actual_bs)

            # Decode and save.
            all_ids = torch.cat(all_output_ids, dim=0)
            # Filter out empty-key entries (e.g. esm placeholder slots with obs_key='')
            # before writing to h5ad obs — empty column names break h5py.
            obs_conditions = {k: v for k, v in conditions.items() if k}
            adatas = []
            for b in range(all_ids.shape[0]):
                token_seq = TokenSequence(
                    token_ids=all_ids[b].tolist(),
                    condition_prefix_length=condition_prefix_length,
                    metadata={"conditions": obs_conditions},
                )
                decoded = tokenizer.decode(token_seq)
                # Copy passthrough columns (non-slot labels the eval needs, e.g.
                # cell_type_coarse / cell_line / drug / dose) verbatim into obs.
                for col in obs_passthrough_columns:
                    if col in task_row.index:
                        decoded.obs[col] = str(task_row[col])
                adatas.append(decoded)

            if adatas:
                merged = ad.concat(adatas, merge="same")
                temporary_path = file_path.with_name(
                    f".{file_path.name}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    merged.write_h5ad(str(temporary_path), compression="gzip")
                    os.replace(temporary_path, file_path)
                finally:
                    temporary_path.unlink(missing_ok=True)
                tqdm.write(f"Saved {len(adatas)} cells: {file_path.name}")

    pbar.close()
    elapsed = time.perf_counter() - t0
    print(f"Generation complete. {total_cells} cells in {elapsed:.1f}s")

    # Merge if all expected batch files are present.
    _merge_if_complete(
        batch_dir, run_dir, full_tasks_df, save_every, total_subtasks,
        filename_columns=filename_columns,
    )


def _expected_batch_files(
    full_tasks_df: pd.DataFrame,
    save_every: int,
    total_subtasks: int,
    filename_columns: Optional[List[str]] = None,
) -> List[str]:
    """Compute the expected batch file names across all subtasks.

    Simulates the file-naming logic for every subtask worker so the
    merge step knows the exact set of files to expect.

    Args:
        full_tasks_df: Complete tasks DataFrame (before subtask split).
        save_every: Periodic save interval.
        total_subtasks: Total number of subtask workers.

    Returns:
        Sorted list of expected file basenames.
    """
    expected: List[str] = []

    for subtask_idx in range(total_subtasks):
        if total_subtasks > 1:
            worker_df = _apply_subtask_split(
                full_tasks_df, subtask_idx + 1, total_subtasks,
            )
        else:
            worker_df = full_tasks_df

        for _, row in worker_df.iterrows():
            n = int(row["n_cells"])
            seed_start = int(row["seed_start"])
            prefix = _make_file_prefix(row, filename_columns)

            if save_every > 0 and n > save_every:
                n_splits = math.ceil(n / save_every)
            else:
                n_splits = 1

            for si in range(n_splits):
                if n_splits > 1:
                    split_row = split_task_cells(row, si, split_size=save_every)
                    if split_row is None:
                        continue
                    s_n = int(split_row["n_cells"])
                    s_seed = int(split_row["seed_start"])
                else:
                    s_n = n
                    s_seed = seed_start
                expected.append(
                    f"{prefix}_seed-{s_seed:09d}-{s_seed + s_n:09d}.h5ad"
                )

    return sorted(expected)


def _merge_if_complete(
    batch_dir: Path,
    run_dir: Path,
    full_tasks_df: pd.DataFrame,
    save_every: int,
    total_subtasks: int,
    filename_columns: Optional[List[str]] = None,
) -> None:
    """Merge batch files into a single h5ad if all expected files are present.

    Computes the expected set of batch files from ``full_tasks_df`` and
    ``save_every``, then checks whether every file exists in ``batch_dir``.
    Only merges when the set is complete — safe for multi-subtask runs
    where earlier workers finish before later ones.

    Args:
        batch_dir: Directory containing per-split h5ad files.
        run_dir: Parent output directory for the merged file.
        full_tasks_df: Complete tasks DataFrame (before subtask split).
        save_every: Periodic save interval.
        total_subtasks: Total number of subtask workers.
    """
    lock_dir = run_dir / ".merge.lockdir"
    merged_path = run_dir / "merged.h5ad"
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if merged_path.exists():
                print(f"Merged file already exists: {merged_path}")
                return
            time.sleep(0.2)
    try:
        if merged_path.exists():
            print(f"Merged file already exists: {merged_path}")
            return

        expected = _expected_batch_files(
            full_tasks_df, save_every, total_subtasks, filename_columns,
        )
        if not expected:
            print("No batch files expected.")
            return

        existing = {f.name for f in batch_dir.glob("*.h5ad")}
        missing = [f for f in expected if f not in existing]
        if missing:
            print(
                f"Merge skipped: {len(missing)}/{len(expected)} batch files missing. "
                f"First missing: {missing[0]}"
            )
            return

        print(f"All {len(expected)} batch files present. Merging...")
        adatas = [ad.read_h5ad(str(batch_dir / filename)) for filename in expected]
        merged = ad.concat(adatas, merge="same")
        merged.obs_names_make_unique()

        temporary_path = run_dir / f".merged.h5ad.{uuid.uuid4().hex}.tmp"
        try:
            merged.write_h5ad(str(temporary_path), compression="gzip")
            os.replace(temporary_path, merged_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"Merged {merged.n_obs} cells → {merged_path}")
    finally:
        lock_dir.rmdir()


def _make_file_prefix(
    task_row: pd.Series,
    filename_columns: Optional[List[str]] = None,
) -> str:
    """Build a filesystem-safe file prefix from condition columns.

    When ``filename_columns`` is given, only those columns are used (in order) —
    e.g. ``[tissue]`` for the tissue / batch-effect evals or
    ``[cell_line_id, drug, drug_concentration_with_unit]`` for Tahoe — so output
    filenames stay short and meaningful. When it is empty/None, every condition
    column except ``n_cells`` / ``seed_start`` is used, preserving the original
    behaviour for VCC / Tahoe-perturbation / PARSE runs (byte-identical).

    Args:
        task_row: One task row whose columns are the condition CSV columns.
        filename_columns: Optional explicit list of columns to name by.

    Returns:
        A filesystem-safe ``"_"``-joined prefix, or ``"uncond"`` if empty.
    """
    cols = filename_columns or [
        c for c in task_row.index if c not in ("n_cells", "seed_start")
    ]
    parts: List[str] = []
    for col in cols:
        if col not in task_row.index:
            continue
        val = str(task_row[col]).replace(" ", "_").replace("/", "-")
        if val:
            parts.append(val)
    return "_".join(parts) if parts else "uncond"


if __name__ == "__main__":
    raise SystemExit(main())
