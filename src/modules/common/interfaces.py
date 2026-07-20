from __future__ import annotations

import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch


@dataclass
class SequenceSample:
    """
    Minimal train-side single-sample structure produced by the dataset.

    Attributes:
        token_ids: Padded token sequence for one sample.
        condition_prefix_length: Number of prefix tokens reserved for conditions.
        seq_length: Effective unpadded sequence length for this sample.
        metadata: Optional non-core sample metadata such as file/source indices.
    """

    token_ids: Any
    condition_prefix_length: int
    seq_length: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SequenceBatch:
    """
    Stable train-side batch structure produced after dataloader collation.

    Attributes:
        token_ids: Batched padded token sequences. This field is expected to be
            tensorized at dataloader return time, typically shaped like `B x L`.
        condition_prefix_length: Batched or broadcast-compatible condition
            prefix lengths. The current mainline still allows list-backed or
            tensor-like representations here.
        seq_length: Batched or broadcast-compatible effective sequence lengths.
            The current mainline still allows list-backed or tensor-like
            representations here.
        metadata: Optional batch metadata or batched sample metadata.
    """

    token_ids: Any
    condition_prefix_length: Any
    seq_length: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointRef:
    """
    Reference to a checkpoint artifact.

    Attributes:
        path: Workflow-selected checkpoint locator.
            Expected forms in the current train workflow include:
            - an explicit checkpoint file path such as
              `workdir/.../step-000123-ckpt.pth`
            - a workflow-managed glob-like pattern such as
              `workdir/vcc/**/step-*-ckpt.pth`
        path_type: Type of the provided path. Do not determine the resolving strategy, which is defined by `resolve_rule`. Expected values include:
            - "full_path": The provided path is a full checkpoint file path, and should be used as-is.
            - "glob_pattern": The provided path is a glob pattern that needs to be resolved to select the actual checkpoint file(s).
        resolve_rule: Rule for resolving the provided path. Expected values include:
            - "exact": Use the provided path as-is, without any resolution or globbing.
            - "latest": If the path is a glob pattern, resolve it to the latest matching.
    """

    path: str
    path_type: str = "full_path"
    resolve_rule: str = "exact"

    def path_exists(self) -> bool:
        """Return whether this checkpoint reference currently resolves to an existing path."""
        if self.path_type == "full_path" and self.resolve_rule == "exact":
            return Path(self.path).is_file()
        if self.path_type == "glob_pattern" and self.resolve_rule == "latest":
            return bool(glob.glob(self.path, recursive=True))
        raise ValueError(
            f"Unsupported checkpoint reference: path_type={self.path_type}, resolve_rule={self.resolve_rule}"
        )

    def resolve_path(self) -> str | None:
        """Resolve this reference to one concrete checkpoint path."""
        if self.path_type == "full_path" and self.resolve_rule == "exact":
            return self.path
        if self.path_type == "glob_pattern" and self.resolve_rule == "latest":
            matches = sorted(glob.glob(self.path, recursive=True))
            if not matches:
                return None
            return max(matches, key=lambda match: Path(match).stat().st_mtime)
        raise ValueError(
            f"Unsupported checkpoint reference: path_type={self.path_type}, resolve_rule={self.resolve_rule}"
        )


@dataclass
class CheckpointState:
    """
    Stable train-state bundle saved through the workflow checkpoint layer.

    Attributes:
        model: Fabric-wrapped runtime model object.
        optimizer: Fabric-wrapped runtime optimizer object, or `None` for
            model-only initialization paths.
        opt_step: Number of completed optimizer steps.
        epoch: Index of the epoch that contains the last completed batch.
        batch_index: Index of the last completed batch inside `epoch`.
        samples_seen: Global number of processed samples up to the last
            completed optimizer step.
        metadata: Optional non-core checkpoint metadata such as the last loss.

    Notes:
        This object only carries the main train-workflow state. It does not
        absorb scheduler-private state, sampler-private resume state, or EMA
        state. Those boundaries are intentionally kept outside the main
        checkpoint contract.
    """

    model: Any = None
    optimizer: Any | None = None
    opt_step: int = 0
    epoch: int = 0
    batch_index: int = -1
    samples_seen: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Expand the checkpoint state into a Fabric-friendly dict."""
        payload: dict[str, Any] = {
            "model": self.model,
            "opt_step": int(self.opt_step),
            "epoch": int(self.epoch),
            "batch_index": int(self.batch_index),
            "samples_seen": int(self.samples_seen),
            "metadata": dict(self.metadata),
        }
        if self.optimizer is not None:
            payload["optimizer"] = self.optimizer
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CheckpointState":
        """Construct one checkpoint state object from a Fabric-loaded dict."""
        return cls(
            model=raw.get("model", None),
            optimizer=raw.get("optimizer", None),
            opt_step=int(raw.get("opt_step", raw.get("step", 0))),
            epoch=int(raw.get("epoch", 0)),
            batch_index=int(raw.get("batch_index", -1)),
            samples_seen=int(raw.get("samples_seen", 0)),
            metadata=dict(raw.get("metadata", {})),
        )
    
    
@dataclass
class ModelInput:
    """Direct model input for the train mainline.

    Treated as a tensor-like batch object by the sampling layer:
    ``to(device)``, ``clone()``, ``len()``, ``shape``/``device`` properties,
    and ``ModelInput.cat([a, b], dim=0)`` (classmethod) all work without
    inspecting metadata contents. The CFG layer is the only place that
    introspects metadata; sampling code (e.g. ``diff_sample``) does not.

    Attributes:
        noisy_token_ids: Preprocessed/noised token inputs actually fed into
            the model. A ``torch.Tensor`` of shape ``(B, L)``. ``dim 0`` is
            the batch dim.
        metadata: Auxiliary model-input metadata.

            Contractual constraints (enforced where reasonable; required by
            ``cat`` semantics):

            - All values MUST be ``torch.Tensor`` (no list/dict/str/etc.).
            - All tensors MUST be batch-aligned: ``tensor.shape[0] == B``,
              matching ``noisy_token_ids.shape[0]``.
            - Other dimensions are field-specific (e.g. ``mask_indices`` is
              ``(B, L)``; ``drug_indices`` / ``dose_values`` are ``(B,)``).
            - Non-tensor or non-batch-aligned auxiliary data should live in
              ``SequenceBatch.metadata``, not here.

            These constraints make ``cat([cond, uncond], dim=0)``
            unambiguous: tensor metadata fields are concatenated along
            their ``dim 0``, mirroring the ``noisy_token_ids`` cat.
    """

    noisy_token_ids: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Tensor-like ergonomics — used by the sampling layer; sampling code
    # (diff_sample, forward_model) treats ModelInput as a unit and does
    # NOT inspect ``metadata`` contents.
    # ------------------------------------------------------------------

    @property
    def shape(self) -> torch.Size:
        """Shape of ``noisy_token_ids`` (typically ``(B, L)``)."""
        return self.noisy_token_ids.shape

    @property
    def device(self) -> torch.device:
        """Device of ``noisy_token_ids``."""
        return self.noisy_token_ids.device

    def __len__(self) -> int:
        """Batch size = ``noisy_token_ids.shape[0]``."""
        return int(self.noisy_token_ids.shape[0])

    def to(self, device) -> "ModelInput":
        """Move all tensor leaves to ``device`` (functional; returns new).

        ``noisy_token_ids`` and any tensor metadata values are moved.
        Per the metadata contract, all metadata values are tensors.

        Args:
            device: Target ``torch.device`` or device string.

        Returns:
            New ``ModelInput`` with all tensors on the target device.
        """
        new_meta = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in self.metadata.items()
        }
        return ModelInput(
            noisy_token_ids=self.noisy_token_ids.to(device),
            metadata=new_meta,
        )

    def clone(self) -> "ModelInput":
        """Deep copy of all tensor leaves.

        Returns:
            New ``ModelInput`` with cloned tensors.
        """
        new_meta = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in self.metadata.items()
        }
        return ModelInput(
            noisy_token_ids=self.noisy_token_ids.clone(),
            metadata=new_meta,
        )

    @classmethod
    def cat(cls, inputs: list["ModelInput"], dim: int = 0) -> "ModelInput":
        """Concatenate a list of ``ModelInput``s along the batch dimension.

        Per the metadata contract, all tensor metadata fields are
        batch-aligned at ``dim 0``. Concatenating ``noisy_token_ids`` and
        each tensor metadata field along ``dim 0`` keeps the output's
        batch structure consistent.

        Strict semantics (fail-fast on contract violations):

        - ``inputs`` must be non-empty.
        - All inputs must share the same metadata key set.
        - Values at each key must be ``torch.Tensor`` and types must match
          across inputs.
        - ``dim`` must be ``0``.

        For non-batch-dim concatenation, construct the result
        ``ModelInput`` manually instead of using ``cat``.

        Args:
            inputs: List of ``ModelInput`` objects to concatenate.
            dim: Concatenation dimension. Must be ``0`` (the batch dim).

        Returns:
            New ``ModelInput`` whose ``noisy_token_ids`` and tensor metadata
            fields are each concatenated along ``dim 0``.

        Raises:
            AssertionError: ``dim != 0``.
            ValueError: ``inputs`` is empty / metadata keys mismatched.
            TypeError: A metadata value is not ``torch.Tensor``, or types
                disagree across inputs at the same key.
        """
        assert dim == 0, (
            "ModelInput.cat only supports dim=0 (batch dimension). "
            "ModelInput.metadata tensors are contractually batch-aligned "
            "(shape[0] == B). Concatenating along other dims would corrupt "
            "(B,)-shaped fields like drug_indices. "
            "If you really need different semantics, construct the result "
            "ModelInput manually instead of using cat."
        )
        if not inputs:
            raise ValueError("ModelInput.cat requires at least one input.")

        first = inputs[0]
        first_keys = set(first.metadata.keys())

        # 1) Validate all inputs share the same metadata key set.
        for i, m in enumerate(inputs[1:], start=1):
            if set(m.metadata.keys()) != first_keys:
                raise ValueError(
                    f"ModelInput.cat: input[{i}] metadata keys differ from "
                    f"input[0]. Got {set(m.metadata.keys())} vs {first_keys}."
                )

        # 2) Concat noisy_token_ids along batch dim.
        new_x = torch.cat([m.noisy_token_ids for m in inputs], dim=dim)

        # 3) Concat each metadata field; enforce tensor-only contract.
        new_meta = {}
        for k in first.metadata:
            vals = [m.metadata[k] for m in inputs]
            ref = vals[0]
            if not isinstance(ref, torch.Tensor):
                raise TypeError(
                    f"ModelInput.cat: metadata[{k!r}] is "
                    f"{type(ref).__name__}, not torch.Tensor. "
                    f"ModelInput.metadata may only contain Tensor values; "
                    f"move non-Tensor data to SequenceBatch.metadata."
                )
            for i, v in enumerate(vals[1:], start=1):
                if not isinstance(v, torch.Tensor):
                    raise TypeError(
                        f"ModelInput.cat: metadata[{k!r}] type mismatch — "
                        f"input[0] is Tensor but input[{i}] is "
                        f"{type(v).__name__}."
                    )
            new_meta[k] = torch.cat(vals, dim=dim)

        return cls(noisy_token_ids=new_x, metadata=new_meta)


@dataclass
class ForwardDiffusionContext:
    """
    Structured train-side forward diffusion outputs kept outside `ModelInput`.

    Attributes:
        mask_indices: Boolean/index tensor aligned with `token_ids`, selecting
            the main diffusion loss positions.
        p_mask: Float tensor aligned with `mask_indices`, representing per-token
            mask probabilities or equivalent weights.
        timesteps: Optional per-sample timestep tensor, typically shaped `B`.
        mask_group_ids: Optional per-sample integer tensor, typically shaped `B`,
            describing which global mask-sharing group each sample belongs to.
        mask_seeds: Optional per-sample integer tensor, typically shaped `B`,
            recording the deterministic seed assigned to each sample.
        metadata: Optional auxiliary forward-diffusion metadata.
    """

    mask_indices: Any
    p_mask: Any
    timesteps: Any = None
    mask_group_ids: Any = None
    mask_seeds: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    """
    Train-side model output.

    Attributes:
        logits: Primary logits tensor used by the current train loss path.
            Shape is ``(B, L, V)`` normally, or ``(N_mask, V)`` when
            ``metadata["is_logits_masked_only"]`` is True.
        metadata: Optional auxiliary outputs or model-side runtime metadata.
            Reserved keys:
            - ``"is_logits_masked_only"`` (bool): When True, ``logits``
              only contains outputs for masked positions (lm_head was
              applied selectively). Loss should use logits directly
              without further mask indexing.
    """

    logits: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LossInputContext:
    """
    Non-model loss context for the train mainline.

    Attributes:
        token_ids: Original padded target token sequence batch, preferably a
            tensor shaped like `B x L`.
        condition_prefix_length: Batched or broadcast-compatible condition
            prefix lengths, preferably tensor-like.
        seq_length: Batched or broadcast-compatible effective sequence lengths,
            preferably tensor-like.
        mask_indices: Boolean/index mask selecting the main diffusion/gene loss
            positions, preferably a tensor aligned with `token_ids`.
        p_mask: Mask probabilities or equivalent weighting factors aligned with
            `mask_indices`, preferably a tensor.
        timesteps: Optional per-sample timestep tensor, preferably shaped `B`.
        mask_group_ids: Optional per-sample integer tensor, preferably shaped `B`.
        mask_seeds: Optional per-sample integer tensor, preferably shaped `B`.
        metadata: Optional auxiliary loss-side metadata kept outside `ModelOutput`.

    Notes:
        The exact alignment convention between `mask_indices`, `p_mask`, `token_ids`
        and `ModelOutput.logits` is defined by the concrete loss implementation.
    """

    token_ids: Any
    condition_prefix_length: Any
    seq_length: Any
    mask_indices: Any = None
    p_mask: Any = None
    timesteps: Any = None
    mask_group_ids: Any = None
    mask_seeds: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LossResult:
    """
    Normalized loss output returned to the workflow.

    Attributes:
        loss: Main scalar loss used for backward propagation.
        metrics: Structured scalar metrics for logging and monitoring.
            The current train workflow expects at least `metrics["total_loss"]`
            to exist as the main logged/scalarized loss value.
        metadata: Optional extra debugging or bookkeeping information.
    """

    loss: Any
    metrics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SamplingRequest:
    """
    Inference-time sampling request.

    Attributes:
        num_samples: Number of samples requested from the sampler.
        base_seed: Base random seed for the sampling job.
        condition_tokens: Optional condition token prefixes supplied to the sampler.
        parameters: Request-scoped runtime parameters that are not fixed sampler config.
        metadata: Optional request metadata.
    """

    num_samples: int
    base_seed: int = 0
    condition_tokens: Sequence["TokenSequence"] = field(default_factory=tuple)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TokenSequence:
    """Lightweight token sequence container used outside the train batch path.

    Attributes:
        token_ids: One generated or decoded token sequence.
        condition_prefix_length: Prefix length carried alongside the sequence
            when relevant.
        metadata: Optional metadata dict. Reserved keys:

            - ``"conditions"``: ``dict[str, str | int | float]`` — condition
              slot values that cannot be decoded from token IDs alone (e.g.
              placeholder slots). Used by ``tokenizer.decode()`` to populate
              obs columns. Keys are slot ``obs_key`` names.

            Other keys may be used for debugging/tracing but should not affect
            encode or decode behavior.
    """

    token_ids: Sequence[int]
    condition_prefix_length: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class InferenceTask:
    """
    Workflow-level inference task bundle.

    Attributes:
        sampling: Sampling request to execute.
        output_dir: Output directory for inference artifacts.
        metadata: Optional task metadata.
    """

    sampling: SamplingRequest
    output_dir: str
    metadata: Mapping[str, Any] = field(default_factory=dict)



@dataclass
class EvaluationInput:
    """
    Normalized evaluator input bundle.

    Attributes:
        predictions: In-memory prediction objects.
        references: In-memory reference objects.
        prediction_path: Optional prediction artifact path.
        reference_path: Optional reference artifact path.
        metadata: Optional evaluation metadata.
    """

    predictions: Sequence[Any] = field(default_factory=tuple)
    references: Sequence[Any] = field(default_factory=tuple)
    prediction_path: str = ""
    reference_path: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """
    Normalized evaluator output bundle.

    Attributes:
        metrics: Scalar or structured metrics produced by evaluation.
        artifacts: Paths to generated evaluation artifacts.
        metadata: Optional extra evaluation metadata.
    """

    metrics: Mapping[str, Any]
    artifacts: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
