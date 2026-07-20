"""Foundation pretrain (S1) tokenizer with batch_id 4-token composite encoding.

Extends :class:`DenseGene1TokTokenizer` to support the foundation condition
prefix, whose only structural novelty is that a single ``batch_id`` obs column
maps to **four** token positions (a composite categorical code) rather than one.

The composite encoding is *pure positional* (decision audit #7): a precomputed
``batch_id_token_mapping.json`` gives each batch_id a 4-tuple
``(source_idx, d2, d3, d4)`` of codebook-local indices. This tokenizer resolves
those local indices to absolute vocab token ids:

    - ``batch_source`` slot (pos 1): ``source_idx`` → ``source_token_vocab``.
    - ``batch_digit`` slot  (pos 2-4): the digit at the tuple index parsed from
      the slot name suffix (``batch_id_t2`` → tuple index 1, ``_t3`` → 2,
      ``_t4`` → 3) → shared ``batch_digit_token_vocab``.

All other slot types (placeholder, categorical, ...) defer to the base class.
The text-embedding placeholders (``cell_description`` / ``perturbation_description``)
emit their placeholder token here; the actual embedding is injected REPLACE-style
at the model side via a per-cell index (handled by the dataset / collate).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata

from .dense_gene1tok_tokenizer import DenseGene1TokTokenizer
from ..condition_formats import ConditionFormat, ConditionSlot


class DenseGene1TokFoundationTokenizer(DenseGene1TokTokenizer):
    """Dense gene-1-token tokenizer for foundation pretrain with batch_id composite.

    Adds two slot types on top of the base tokenizer:

        - ``batch_source``: encodes ``batch_id`` → source codebook token.
        - ``batch_digit``: encodes one base-100 digit of the batch_id's
          intra-source running index → shared digit codebook token.

    Attributes:
        batch_id_mapping_file: Resolved path to the batch_id → 4-token mapping.
        _batch_id_token_map: Dict ``batch_id_str -> [source_idx, d2, d3, d4]``.
        _batch_source_vocab: Ordered source names; index = ``source_idx``.
    """

    def __init__(
        self,
        *,
        vocab_file: str,
        condition_format: ConditionFormat,
        batch_id_mapping_file: str,
    ) -> None:
        """Initialize the foundation tokenizer and load the batch_id mapping.

        Args:
            vocab_file: Path to the foundation vocab JSON.
            condition_format: 18-slot foundation condition prefix descriptor.
            batch_id_mapping_file: Path to ``batch_id_token_mapping.json``
                produced by ``generate_batch_id_token_mapping.py``.

        Raises:
            FileNotFoundError: If *batch_id_mapping_file* cannot be located.
            ValueError: If the mapping or required codebooks are malformed /
                missing (e.g. ``source_token_vocab`` / ``batch_digit_token_vocab``).
        """
        # Base class loads vocab + validates slots/codebooks (the new slot types
        # carry valid codebooks, so validation passes unchanged).
        super().__init__(vocab_file=vocab_file, condition_format=condition_format)

        # --- Load the batch_id → 4-token mapping ---
        resolved = Path(batch_id_mapping_file)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"batch_id mapping file not found: {batch_id_mapping_file}"
            )
        self.batch_id_mapping_file = str(resolved)
        with open(resolved, "r", encoding="utf-8") as handle:
            mapping_data = json.load(handle)
        self._batch_id_token_map: dict[str, list[int]] = mapping_data["mapping"]
        self._batch_source_vocab: list[str] = list(mapping_data["source_vocab"])

        # --- Validate the codebooks the composite encoding depends on ---
        for required_cb in ("source_token_vocab", "batch_digit_token_vocab"):
            if required_cb not in self._codebooks:
                raise ValueError(
                    f"Foundation tokenizer requires codebook '{required_cb}' "
                    f"in the vocab file but it is missing."
                )

    # ------------------------------------------------------------------
    # batch_id composite encoding
    # ------------------------------------------------------------------

    def _batch_tuple(self, batch_id: str) -> list[int]:
        """Look up the 4-token composite tuple for a batch_id string.

        Args:
            batch_id: Raw ``obs.batch_id`` value.

        Returns:
            ``[source_idx, d2, d3, d4]`` codebook-local indices.

        Raises:
            KeyError: If *batch_id* is absent from the mapping (Task 1 must
                guarantee 100% coverage of all split batch_ids).
        """
        try:
            return self._batch_id_token_map[batch_id]
        except KeyError as exc:
            raise KeyError(
                f"batch_id {batch_id!r} not found in batch_id_token_mapping. "
                f"The mapping must cover every batch_id present in the data."
            ) from exc

    @staticmethod
    def _digit_tuple_index(slot_name: str) -> int:
        """Parse the 4-tuple index a ``batch_digit`` slot encodes from its name.

        Slot naming convention ``batch_id_t{N}`` maps to tuple index ``N-1``
        (``batch_id_t2`` → 1, ``batch_id_t3`` → 2, ``batch_id_t4`` → 3).

        Args:
            slot_name: The ConditionSlot name (e.g. ``"batch_id_t2"``).

        Returns:
            The integer index into the batch tuple.

        Raises:
            ValueError: If the slot name does not match the ``*_t{N}`` pattern.
        """
        try:
            n = int(slot_name.rsplit("_t", 1)[-1])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"batch_digit slot name {slot_name!r} must end with '_t<N>' "
                f"(e.g. 'batch_id_t2')."
            ) from exc
        return n - 1

    def _encode_slot_value(self, slot: ConditionSlot, value: str) -> int:
        """Encode a slot value, dispatching the two batch_id composite types.

        Args:
            slot: The ``ConditionSlot`` being encoded.
            value: Resolved string value. For batch slots this is the raw
                ``batch_id`` string.

        Returns:
            Absolute vocab token id for this slot position.
        """
        # --- batch_id token1: source codebook ---
        if slot.slot_type == "batch_source":
            source_idx = self._batch_tuple(value)[0]
            source_name = self._batch_source_vocab[source_idx]
            return int(self._codebooks["source_token_vocab"][source_name])

        # --- batch_id token2-4: shared digit codebook ---
        if slot.slot_type == "batch_digit":
            tuple_index = self._digit_tuple_index(slot.name)
            digit = self._batch_tuple(value)[tuple_index]
            return int(self._codebooks["batch_digit_token_vocab"][str(digit)])

        # --- All other slot types: base behaviour ---
        return super()._encode_slot_value(slot, value)

    def _decode_slot_value(self, slot: ConditionSlot, token_id: int) -> str:
        """Decode a token id, handling the batch composite slot types.

        ``batch_source`` decodes to the source name; ``batch_digit`` decodes to
        the digit string. Full ``batch_id`` reconstruction is not possible from
        a single token, so callers should rely on ``metadata["conditions"]``
        (handled in :meth:`decode`). Other slot types defer to the base class.

        Args:
            slot: The ``ConditionSlot`` being decoded.
            token_id: Integer token id at the slot's prefix position.

        Returns:
            Decoded per-token string value.
        """
        if slot.slot_type == "batch_source":
            for name, tid in self._codebooks["source_token_vocab"].items():
                if int(tid) == token_id:
                    return name
            return slot.override_value
        if slot.slot_type == "batch_digit":
            for digit, tid in self._codebooks["batch_digit_token_vocab"].items():
                if int(tid) == token_id:
                    return digit
            return slot.override_value
        return super()._decode_slot_value(slot, token_id)

    def decode(self, sequence: Any) -> anndata.AnnData:
        """Decode a token sequence to AnnData, restoring batch_id from metadata.

        The 4-token batch_id composite cannot be losslessly inverted per-slot
        (all four share ``obs_key='batch_id'``), so the real batch_id is taken
        from ``sequence.metadata['conditions']`` when present. Reserved dummy
        obs columns (``_reserved_*``) are dropped for a clean AnnData.

        Args:
            sequence: A ``TokenSequence`` (same contract as the base decode).

        Returns:
            A 1-row ``anndata.AnnData`` with expression counts and condition
            obs columns (batch_id restored from metadata; reserved columns
            removed).
        """
        adata = super().decode(sequence)

        # --- Restore batch_id from metadata (composite is non-invertible per-slot) ---
        conditions = sequence.metadata.get("conditions", {})
        if "batch_id" in conditions:
            adata.obs["batch_id"] = str(conditions["batch_id"])

        # --- Drop reserved dummy obs columns for cleanliness ---
        reserved_cols = [c for c in adata.obs.columns if c.startswith("_reserved_")]
        if reserved_cols:
            adata.obs = adata.obs.drop(columns=reserved_cols)

        return adata
