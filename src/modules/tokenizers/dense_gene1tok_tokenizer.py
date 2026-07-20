"""General dense gene-1-token tokenizer driven by a declarative ConditionFormat.

This tokenizer derives its condition prefix layout entirely from a
``ConditionFormat`` descriptor.  Changing the YAML is sufficient to adapt to
different condition schemas without writing new Python code.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Mapping

import anndata
import numpy as np
import pandas as pd
import scipy.sparse

from ..common import TokenSequence
from .base import Tokenizer
from ..condition_formats import ConditionFormat, ConditionSlot


class DenseGene1TokTokenizer(Tokenizer):
    """General dense gene-1-token tokenizer driven by a declarative ConditionFormat.

    Each gene in the panel is represented by a single count token placed at a
    fixed dense index.  A condition prefix of fixed length precedes the
    expression tokens.  The prefix layout is fully determined by the
    ``ConditionFormat`` passed at construction time.

    Attributes:
        vocab_file: Resolved path to the JSON vocab file on disk.
        condition_format: The ``ConditionFormat`` descriptor that governs
            prefix layout and slot encoding.
        condition_prefix_length: Number of leading token positions reserved
            for condition encoding.
        seq_length: Total fixed sequence length (``condition_prefix_length +
            max_gene_count``).
        vocab_size: Total number of distinct token ids across all token
            categories.
        eos_id: Token id for the ``</s>`` end-of-sequence / padding token.
        unknown_gene_id: Token id for the ``<unk_gene>`` fallback token.
        overflow_token_id: Token id emitted when a count exceeds the
            maximum representable value.
        target_gene_token_id_offset: Token-id offset used to recover the
            dense gene index from a target-gene condition token.
        gene_names_ordered: Ordered gene panel; index equals dense position.
        gene_name_to_dense_idx: Reverse lookup from gene name to dense index.
        control_label: Canonical string label for control samples
            (e.g. ``"non-targeting"``, ``"DMSO"``, ``"untreated"``).
    """

    def __init__(
        self,
        *,
        vocab_file: str,
        condition_format: ConditionFormat,
    ) -> None:
        """Initialize the tokenizer and load vocab metadata.

        Args:
            vocab_file: Path to the JSON vocab file.  Resolved relative to
                the current working directory first; if not found, falls back
                to ``archive/legacy/<vocab_file>`` inside the repo root.
            condition_format: Declarative description of the condition prefix
                layout.  Determines prefix length, slot types, and codebook
                lookups.

        Raises:
            FileNotFoundError: If *vocab_file* cannot be located at either
                the direct path or the legacy fallback path.
            ValueError: If the vocab file is missing required keys or tokens.
            NotImplementedError: If any slot has ``width != 1``.
        """
        # Resolve vocab file path, with legacy archive fallback
        resolved_vocab_file = Path(vocab_file)
        if not resolved_vocab_file.is_file():
            repo_root_candidate = Path(__file__).resolve().parents[3]
            legacy_candidate = repo_root_candidate / "archive" / "legacy" / vocab_file
            if legacy_candidate.is_file():
                resolved_vocab_file = legacy_candidate
            else:
                raise FileNotFoundError(f"Tokenizer vocab file not found: {vocab_file}")

        self.vocab_file = str(resolved_vocab_file)
        self.condition_format = condition_format
        self.condition_prefix_length = condition_format.prefix_length

        self._codebooks: dict[str, Any] = {}
        self._load_vocab_metadata()

    # ------------------------------------------------------------------
    # Vocab loading
    # ------------------------------------------------------------------

    def _load_vocab_metadata(self) -> None:
        """Parse the vocab JSON file and populate all tokenizer state.

        Reads the vocab file and initialises every attribute needed for
        encoding and decoding.  Called once from ``__init__``; not intended
        to be called again after construction.

        The vocab JSON is expected to contain:
            - ``special_token_vocab``: mapping of special token names to ids.
            - ``count_to_token_id_info``: count-encoding parameters and rules.
            - One entry per slot ``codebook_key`` referenced in the
              ``ConditionFormat`` (e.g. ``"cell_type_token_vocab"``).
            - Optionally ``special_control_vocab`` (or legacy
              ``special_target_gene_vocab``): mapping for control labels.

        Raises:
            ValueError: If required tokens or codebook keys are missing.
            NotImplementedError: If any slot declares ``width != 1``.
        """
        # Load raw vocab JSON
        with open(self.vocab_file, "r", encoding="utf-8") as handle:
            vocab_data = json.load(handle)

        # Unpack top-level vocab sections
        self.special_token_vocab: dict[str, int] = dict(vocab_data["special_token_vocab"])
        self.count_to_token_id_info: dict[str, Any] = dict(vocab_data["count_to_token_id_info"])

        # Load per-slot codebooks declared in the ConditionFormat
        for slot in self.condition_format.slots:
            if slot.codebook_key and slot.codebook_key in vocab_data:
                self._codebooks[slot.codebook_key] = vocab_data[slot.codebook_key]

        # Store auxiliary index maps if present (e.g. drug_index_map for perturbation tasks)
        self.drug_index_map: dict[str, int] = dict(vocab_data.get("drug_index_map", {}))

        # Validate that all non-gene-index codebook keys were found
        for slot in self.condition_format.slots:
            if (
                slot.codebook_key
                and slot.slot_type != "gene_index"
                and slot.codebook_key not in self._codebooks
            ):
                raise ValueError(
                    f"Slot '{slot.name}' requires codebook '{slot.codebook_key}' "
                    f"but it is missing from the vocab file."
                )

        # Resolve required special token ids
        self.eos_id = int(self.special_token_vocab["</s>"])
        self.bookend_token_id = int(self.special_token_vocab[self.condition_format.bookend_token])
        self._unknown_gene_token_name = "<unk_gene>"
        if self._unknown_gene_token_name not in self.special_token_vocab:
            raise ValueError(
                f"Vocab file missing required special token '{self._unknown_gene_token_name}'."
            )
        self.unknown_gene_id = int(self.special_token_vocab[self._unknown_gene_token_name])

        # Unpack count encoding parameters
        self.overflow_token_id = int(self.count_to_token_id_info["overflow_token_id"])
        self.min_count_token_id = int(self.count_to_token_id_info["min_token_id"])
        self.max_count_token_id = int(self.count_to_token_id_info["max_token_id"])
        self.count_mapping_rules = tuple(self.count_to_token_id_info["count_mapping_rules"])

        # Load gene panel from gene-index codebook, top-level keys, or empty
        gene_index_cb_key = self._gene_index_codebook_key()
        if gene_index_cb_key is not None:
            if gene_index_cb_key not in self._codebooks:
                raise ValueError(
                    f"Gene-index slot requires codebook '{gene_index_cb_key}' "
                    f"but it is missing from the vocab file."
                )
            gene_info = self._codebooks[gene_index_cb_key]
            offset_key = (
                "gene_index_offset" if "gene_index_offset" in gene_info
                else "target_gene_token_id_offset"
            )
            if offset_key not in gene_info:
                raise ValueError(
                    f"Gene-index codebook '{gene_index_cb_key}' missing "
                    f"required key 'gene_index_offset' (or legacy "
                    f"'target_gene_token_id_offset')."
                )
            self.gene_index_token_id_offset: int = int(gene_info[offset_key])
            self.max_gene_count = int(gene_info["max_gene_count"])
            self.gene_names_ordered: tuple[str, ...] = tuple(gene_info["gene_names_ordered"])
            self._gene_tokens_in_vocab = True
        elif "max_gene_count" in vocab_data and "gene_names_ordered" in vocab_data:
            self.gene_index_token_id_offset: int = 0  # type: ignore[no-redef]
            self.max_gene_count = int(vocab_data["max_gene_count"])
            self.gene_names_ordered: tuple[str, ...] = tuple(  # type: ignore[no-redef]
                str(g) for g in vocab_data["gene_names_ordered"]
            )
            self._gene_tokens_in_vocab = False
        else:
            self.gene_index_token_id_offset: int = 0  # type: ignore[no-redef]
            self.max_gene_count = 0
            self.gene_names_ordered: tuple[str, ...] = ()  # type: ignore[no-redef]
            self._gene_tokens_in_vocab = False
        self.target_gene_token_id_offset = self.gene_index_token_id_offset
        self.gene_name_to_dense_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(self.gene_names_ordered)
        }

        # Load target gene vocab: compact gene→token_id mapping that
        # decouples target gene encoding from expression body gene order.
        # When present, only these genes can appear as gene_index condition
        # values. When absent, all genes in gene_names_ordered are valid.
        self._target_gene_vocab: dict[str, int] = dict(
            vocab_data.get("target_gene_vocab", {})
        )

        # Build unified target gene → token_id lookup
        if self._target_gene_vocab:
            self._target_gene_to_token_id = dict(self._target_gene_vocab)
        elif self._gene_tokens_in_vocab:
            self._target_gene_to_token_id = {
                name: self.gene_index_token_id_offset + idx
                for idx, name in enumerate(self.gene_names_ordered)
            }
        else:
            self._target_gene_to_token_id = {}

        # Resolve control label and its special token id
        ci = self.condition_format.control_indicator
        default_control_label = ci.control_value if ci is not None else ""
        special_control_vocab: dict[str, int] = dict(
            vocab_data.get("special_control_vocab")
            or vocab_data.get("special_target_gene_vocab")
            or {}
        )
        self.control_label = str(next(iter(special_control_vocab), default_control_label))
        self.control_index_id = int(special_control_vocab.get(self.control_label, self.unknown_gene_id))
        self._special_control_count = len(special_control_vocab)

        # Derive sequence and vocab dimensions
        self.seq_length = self.condition_prefix_length + self.max_gene_count

        self._vocab_size = self._compute_vocab_size(vocab_data)

        # Build auxiliary lookup structures
        self._build_compat_fields()
        self._prefix_position_map = self.condition_format.slot_position_map

        for slot in self.condition_format.slots:
            if slot.width != 1:
                raise NotImplementedError(
                    f"Slot '{slot.name}' has width={slot.width}; only width=1 is supported."
                )

    def _compute_vocab_size(self, vocab_data: dict) -> int:
        """Sum token counts across **all** ID sources in the vocab JSON.

        Unlike the previous formula that only counted codebooks referenced
        by the current ``ConditionFormat``, this scans every token-bearing
        section in the vocab file so the embedding table is always large
        enough even when some codebooks are unused by the active format.

        Counted sources:

        1. Every ``*_token_vocab`` codebook dict in the JSON.
        2. Count token range (``max_token_id - min_token_id + 1``).
        3. ``special_control_vocab`` or legacy ``special_target_gene_vocab``.
        4. Gene-index token range (when present in the vocab).

        Args:
            vocab_data: Raw dict loaded from the vocab JSON file.

        Returns:
            Total token count across all sources.
        """
        vocab_size = 0

        # 1. All *_token_vocab codebook dicts (skip special_token_vocab)
        for key, value in vocab_data.items():
            if key.endswith("_token_vocab") and isinstance(value, dict):
                vocab_size += len(value)

        # 2. Count token range
        vocab_size += self.max_count_token_id - self.min_count_token_id + 1

        # 3. Special control vocab (may use legacy key name)
        for key in ("special_control_vocab", "special_target_gene_vocab"):
            val = vocab_data.get(key)
            if isinstance(val, dict):
                vocab_size += len(val)

        # 4. Gene-index token range (target_gene_vocab overrides full gene panel)
        #    Count unique token IDs, not dict keys, to handle alias mappings
        #    where multiple gene names (e.g. IRE1/ERN1) share the same ID.
        if self._target_gene_vocab:
            vocab_size += len(set(self._target_gene_vocab.values()))
        elif self._gene_tokens_in_vocab and self.max_gene_count > 0:
            vocab_size += self.max_gene_count

        return vocab_size

    def _gene_index_codebook_key(self) -> str | None:
        """Return the codebook key for the first gene-index slot, or None.

        The gene panel and token-id offset are stored under this key in the
        vocab JSON.  Returns ``None`` when the ``ConditionFormat`` contains
        no ``gene_index`` slot (e.g. in condition-only configurations).

        Returns:
            Codebook key string, or ``None`` if no ``gene_index``
            slot exists.
        """
        for slot in self.condition_format.slots:
            if slot.slot_type == "gene_index" and slot.codebook_key:
                return slot.codebook_key
        return None

    def _build_compat_fields(self) -> None:
        """Derive VCC-compat Protocol fields from ConditionFormat slots.

        Populates a set of named attributes expected by the VCC training
        workflow and downstream evaluation code.  These fields mirror the
        interface exposed by the legacy ``DenseGene1TokVCCTokenizer`` so
        that both tokenizers are interchangeable where the workflow uses
        these attributes directly.

        Attributes set:
            condition_schema: Always ``"general"`` for this tokenizer.
            target_gene_2token_width: ``None`` (2-token target not supported).
            condition_cell_type_idx: Prefix position of the ``cell_type`` slot,
                or ``-1`` if absent.
            condition_target_gene_high_idx: Prefix position of the
                ``target_gene`` slot, or ``-1`` if absent.
            condition_target_gene_low_idx: Always ``-1`` (2-token layout not
                used).
            condition_cell_type_obs_key: ``obs_key`` of the ``cell_type``
                slot, or empty string.
            condition_target_obs_key: ``obs_key`` of the ``target_gene``
                slot, or empty string.
            condition_target_is_gene: ``True`` when the target slot uses
                ``slot_type == "gene_index"``.
            default_cell_type_value: Override value for the ``cell_type`` slot
                (empty when data-driven).
            default_obs_values: Mapping of obs key to override value for
                every slot (empty values mean data-driven).
            required_obs_keys: Tuple of all primary obs column names.
        """
        self.condition_schema = "general"
        self.target_gene_2token_width = None

        prefix_positions = self.condition_format.slot_position_map

        cell_type_slot = self.condition_format.slot_by_name("cell_type")
        target_gene_slot = self.condition_format.slot_by_name("target_gene")

        self.condition_cell_type_idx = prefix_positions.get("cell_type", -1)
        self.condition_target_gene_high_idx = prefix_positions.get("target_gene", -1)
        self.condition_target_gene_low_idx = -1
        self.condition_cell_type_obs_key = cell_type_slot.obs_key if cell_type_slot else ""
        self.condition_target_obs_key = target_gene_slot.obs_key if target_gene_slot else ""
        self.condition_target_is_gene = target_gene_slot is not None and target_gene_slot.slot_type == "gene_index"

        self.default_cell_type_value = cell_type_slot.override_value if cell_type_slot else ""
        self.default_obs_values: Mapping[str, Any] = {
            slot.obs_key: slot.override_value for slot in self.condition_format.slots
        }
        # Required obs keys = slots whose value must come from the data. Slots
        # carrying an ``override_value`` (e.g. foundation ``_reserved_*`` slots
        # forced to ``"<reserved>"``) are filled by the ConditionFormat itself
        # at encode time and must NOT be demanded as condition-CSV columns.
        # ``dict.fromkeys`` dedupes obs_keys shared by multiple slots (e.g. the
        # four foundation ``batch_id`` composite slots) while preserving order.
        self.required_obs_keys: tuple[str, ...] = tuple(
            dict.fromkeys(
                slot.obs_key
                for slot in self.condition_format.slots
                if slot.obs_key and not slot.override_value
            )
        )

    @property
    def valid_target_gene_names(self) -> frozenset[str]:
        """Return the set of gene names accepted as target_gene condition values.

        Derived from ``_target_gene_to_token_id`` — which is built from
        ``target_gene_vocab`` (compact) or ``gene_names_ordered`` (full panel).
        """
        return frozenset(self._target_gene_to_token_id.keys())

    # ------------------------------------------------------------------
    # Slot encoding / decoding helpers
    # ------------------------------------------------------------------

    def _encode_slot_value(self, slot: ConditionSlot, value: str) -> int:
        """Encode a single slot value to its token id.

        Dispatches to the appropriate encoding strategy based on
        ``slot.slot_type``.

        Args:
            slot: The ``ConditionSlot`` descriptor for the position being
                encoded.
            value: String value to encode (already resolved from obs or
                default).

        Returns:
            Token id for the encoded value.

        Raises:
            ValueError: If ``slot_type == "categorical"`` and *value* is not
                in the codebook (and no override fallback is available), or
                if ``slot_type`` is not one of ``"special_token"``,
                ``"categorical"``, ``"gene_index"``, ``"placeholder"``.
        """
        if slot.slot_type == "special_token":
            if value not in self.special_token_vocab:
                raise ValueError(
                    f"Special-token slot '{slot.name}': value '{value}' "
                    f"not in special_token_vocab."
                )
            return int(self.special_token_vocab[value])

        if slot.slot_type == "categorical":
            codebook = self._codebooks.get(slot.codebook_key, {})
            if isinstance(codebook, dict) and value in codebook:
                return int(codebook[value])
            if slot.override_value and isinstance(codebook, dict) and slot.override_value in codebook:
                warnings.warn(
                    f"Categorical slot '{slot.name}': value '{value}' not in "
                    f"codebook, falling back to override '{slot.override_value}'.",
                    stacklevel=2,
                )
                return int(codebook[slot.override_value])
            raise ValueError(
                f"Categorical slot '{slot.name}': value '{value}' not in codebook"
                f"{f', override {slot.override_value!r} also missing' if slot.override_value else ''}."
            )

        if slot.slot_type == "gene_index":
            return self._encode_gene_index_value(value)

        if slot.slot_type == "placeholder":
            codebook = self._codebooks.get(slot.codebook_key, {})
            assert len(codebook) == 1, (
                f"Placeholder slot '{slot.name}' codebook must have exactly 1 entry, got {len(codebook)}"
            )
            return int(next(iter(codebook.values())))

        raise ValueError(f"Unknown slot_type: {slot.slot_type}")

    def _encode_gene_index_value(self, gene_name: str) -> int:
        """Encode a gene name to its gene-index token id.

        Resolves control labels case-insensitively.  Looks up the gene
        name in ``_target_gene_to_token_id`` (built at init from either
        ``target_gene_vocab`` or ``gene_names_ordered``).  Returns
        ``unknown_gene_id`` for any gene name not in the lookup.

        Args:
            gene_name: Gene name string (e.g. ``"BRCA1"``) or a special
                label such as ``"non-targeting"`` or ``"<unk_gene>"``.

        Returns:
            Token id for the gene name.
        """
        if gene_name == self._unknown_gene_token_name:
            return self.unknown_gene_id
        if gene_name.lower() == self.control_label.lower():
            return self.control_index_id
        token_id = self._target_gene_to_token_id.get(gene_name)
        if token_id is not None:
            return token_id
        return self.unknown_gene_id

    def _decode_slot_value(self, slot: ConditionSlot, token_id: int) -> str:
        """Decode a token id back to a string condition value for a given slot.

        Dispatches to the appropriate decoding strategy based on
        ``slot.slot_type``.  Falls back to ``slot.override_value`` when the
        token id is not found in the relevant lookup.

        Args:
            slot: The ``ConditionSlot`` descriptor for the prefix position
                being decoded.
            token_id: Integer token id to look up.

        Returns:
            Decoded string value, or ``slot.override_value`` if the id is
            unrecognised.
        """
        if slot.slot_type == "special_token":
            for name, tid in self.special_token_vocab.items():
                if int(tid) == token_id:
                    return name
            return slot.override_value

        if slot.slot_type == "categorical":
            codebook = self._codebooks.get(slot.codebook_key, {})
            if isinstance(codebook, dict):
                for name, tid in codebook.items():
                    if int(tid) == token_id:
                        return name
            return slot.override_value

        if slot.slot_type == "gene_index":
            return self._decode_gene_index_value(token_id)

        return slot.override_value

    def _decode_gene_index_value(self, token_id: int) -> str:
        """Decode a gene-index token id back to a gene name string.

        Inverse of ``_encode_gene_index_value``.  When
        ``target_gene_vocab`` is present, reverse-looks up via the compact
        mapping; otherwise falls back to ``gene_names_ordered`` indexing.

        Args:
            token_id: Integer token id to decode.

        Returns:
            Gene name string, ``control_label`` for the control token,
            or ``"<unk_gene>"`` if the id is out of range.
        """
        if token_id == self.unknown_gene_id:
            return self._unknown_gene_token_name
        if token_id == self.control_index_id:
            return self.control_label
        # When using compact target_gene_vocab, reverse-lookup directly.
        if self._target_gene_vocab:
            for name, tid in self._target_gene_to_token_id.items():
                if tid == token_id:
                    return name
            return self._unknown_gene_token_name
        # Fallback: dense_idx into gene_names_ordered (legacy full-panel encoding).
        dense_idx = token_id - self.gene_index_token_id_offset
        if 0 <= dense_idx < len(self.gene_names_ordered):
            return self.gene_names_ordered[dense_idx]
        return self._unknown_gene_token_name

    # ------------------------------------------------------------------
    # Count encoding
    # ------------------------------------------------------------------

    def _encode_count_token(self, count: int) -> int:
        """Encode a raw integer expression count to its count token id.

        Clamps negative counts to zero before applying encoding rules.

        Args:
            count: Raw integer UMI/read count.  Negative values are treated
                as zero.

        Returns:
            Token id from the count encoding scheme (identity or
            2-important-digits mapping, depending on the rule range).
        """
        if count <= 0:
            return self._encode_count_token_from_rules(0)
        return self._encode_count_token_from_rules(int(count))

    def _encode_count_token_from_rules(self, count: int) -> int:
        """Apply count mapping rules to convert an integer count to a token id.

        Iterates through the ordered ``count_mapping_rules`` from the vocab
        file.  Each rule covers a count range and specifies a mapping method:

        - ``"identity"``: each count in the range maps to a distinct token id
          (``base + (count - rule_min)``).
        - ``"2_important_digits"``: encodes only the two most significant
          digits, grouping counts by order of magnitude.

        Args:
            count: Non-negative integer count.  Values below ``min_count``
                map to ``min_count_token_id``; values above ``max_count``
                map to ``overflow_token_id``.

        Returns:
            Token id for the given count.

        Raises:
            ValueError: If the count falls within the valid range but is not
                covered by any rule, or if a rule specifies an unknown
                ``mapping_method``.
        """
        min_count = int(self.count_to_token_id_info["min_count"])
        max_count = int(self.count_to_token_id_info["max_count"])
        if count < min_count:
            return self.min_count_token_id
        if count > max_count:
            return self.overflow_token_id
        for rule in self.count_mapping_rules:
            rule_min = int(rule["min_count"])
            rule_max = int(rule["max_count"])
            if rule_min <= count <= rule_max:
                base_token = int(rule["min_token_id"])
                mapping_method = str(rule["mapping_method"])
                if mapping_method == "identity":
                    return base_token + (count - rule_min)
                if mapping_method == "2_important_digits":
                    length = len(str(count))
                    divisor = 10 ** (length - 2)
                    return base_token + (length - 3) * 90 + (count // divisor - 10)
                raise ValueError(f"Unsupported count mapping method: {mapping_method}")
        raise ValueError(f"Count {count} is not covered by tokenizer count mapping rules.")

    def _decode_count_token(self, token_id: int) -> int:
        """Decode a count token id back to the lowerbound integer expression count.

        Inverse of ``_encode_count_token_from_rules``.  The overflow token
        decodes to ``max_count``; any unrecognised token id returns ``0``.

        Args:
            token_id: Integer token id from the count region of the vocab.

        Returns:
            Decoded integer count.  Returns ``0`` if *token_id* does not
            match any rule range.
        """
        if token_id == self.overflow_token_id:
            return int(self.count_to_token_id_info["max_count"])
        for rule in self.count_mapping_rules:
            rule_min_tid = int(rule["min_token_id"])
            rule_max_tid = int(rule["max_token_id"])
            if rule_min_tid <= token_id <= rule_max_tid:
                mapping_method = str(rule["mapping_method"])
                if mapping_method == "identity":
                    return int(rule["min_count"]) + (token_id - rule_min_tid)
                if mapping_method == "2_important_digits":
                    offset = token_id - rule_min_tid
                    length = offset // 90 + 3
                    divisor = 10 ** (length - 2)
                    return (offset - (length - 3) * 90 + 10) * divisor
        return 0

    # ------------------------------------------------------------------
    # Public Protocol methods
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(
        self,
        adata: Any,
        row_index: int = 0,
        *,
        gene_names: tuple[str, ...] | None = None,
    ) -> TokenSequence:
        """Encode one AnnData sample into a complete token sequence (prefix + expression).

        Reads condition values from ``adata.obs.iloc[row_index]`` for each
        slot defined in the ``ConditionFormat``, falling back to alias obs
        keys and then to slot defaults.  Gene expression is read from
        ``adata.X[row_index]``.

        Args:
            adata: An ``anndata.AnnData`` object.  Must have at least
                ``row_index + 1`` rows.
            row_index: Row to encode within *adata*.  Defaults to ``0``.
            gene_names: Optional pre-resolved gene name tuple aligned with
                the columns of ``adata.X``.  When ``None``, derived from
                ``adata.var_names``.

        Returns:
            A ``TokenSequence`` with ``token_ids`` of fixed length
            ``seq_length`` (``condition_prefix_length + max_gene_count``) and
            ``condition_prefix_length`` set accordingly.
        """
        obs = adata.obs.iloc[row_index]
        conditions = self.condition_format.resolve_obs_values(obs)
        prefix_tokens = self.encode_condition(conditions)
        if gene_names is None:
            gene_names = tuple(str(g) for g in adata.var_names.tolist())
        dense_tokens = self.encode_dense_counts(row=adata.X[row_index], gene_names=gene_names)
        return TokenSequence(
            token_ids=tuple(prefix_tokens + dense_tokens),
            condition_prefix_length=self.condition_prefix_length,
        )

    def encode_condition(self, conditions: Mapping[str, str]) -> list[int]:
        """Encode condition values into a prefix token list.

        Used internally by ``encode()`` and directly by inference workflows
        that need to construct condition prefixes without expression data.

        **Callers must pass pre-normalized condition values.**  In particular,
        control labels should already use the canonical form (e.g.
        ``"non-targeting"``).  Use ``ConditionFormat.normalize_obs_dataframe``
        at the dataset level to ensure obs values are canonical before
        encoding.

        Args:
            conditions: Mapping from slot ``obs_key`` to its string value.
                All slots defined in the ``ConditionFormat`` must be present
                and non-empty (e.g. ``{"cell_type": "H1",
                "target_gene": "BRCA1"}``).

        Returns:
            List of ``condition_prefix_length`` integer token ids, with
            bookend positions filled by ``bookend_token_id``.

        Raises:
            KeyError: If any required slot obs_key is absent or empty in
                *conditions*.
        """
        resolved: dict[str, str] = {}
        for slot in self.condition_format.slots:
            value = conditions.get(slot.obs_key)
            if value in (None, ""):
                raise KeyError(
                    f"encode_condition: missing or empty value for slot "
                    f"'{slot.name}' (obs_key='{slot.obs_key}'). "
                    f"Received keys: {list(conditions.keys())}"
                )
            resolved[slot.obs_key] = str(value)
        return self.condition_format.build_prefix(
            resolved, self.bookend_token_id, self._encode_slot_value
        )

    def decode(self, sequence: TokenSequence) -> anndata.AnnData:
        """Decode a token sequence into a 1-row AnnData.

        Reconstructs condition metadata from the condition prefix and
        expression counts from the gene expression suffix.  Genes with a
        zero-count token are omitted from the sparse matrix; all other
        positions are decoded and stored.

        Args:
            sequence: A ``TokenSequence`` with at least
                ``condition_prefix_length`` token ids.  Additional tokens
                up to ``max_gene_count`` are decoded as expression counts.

        Returns:
            A 1-row ``anndata.AnnData`` with:

            - ``X``: sparse CSR matrix of shape ``(1, max_gene_count)``
              containing decoded integer counts.
            - ``obs``: ``DataFrame`` with one column per slot obs_key
              (e.g. ``cell_type``, ``target_gene``).
            - ``var``: ``DataFrame`` indexed by ``gene_names_ordered``.

        Raises:
            ValueError: If ``sequence.token_ids`` is shorter than
                ``condition_prefix_length``.
        """
        token_ids = list(sequence.token_ids)
        if len(token_ids) < self.condition_prefix_length:
            raise ValueError(
                f"Token sequence too short: expected at least "
                f"{self.condition_prefix_length}, got {len(token_ids)}."
            )
        obs = self.condition_format.decode_prefix(token_ids, self._decode_slot_value)

        # Apply condition overrides for placeholder slots that cannot be
        # decoded from token IDs (the real values are carried in metadata).
        conditions = sequence.metadata.get("conditions", {})
        if conditions:
            for slot in self.condition_format.slots:
                if slot.slot_type == "placeholder" and slot.obs_key in conditions:
                    obs[slot.obs_key] = str(conditions[slot.obs_key])

        count_tokens = token_ids[self.condition_prefix_length:]
        zero_id = self._encode_count_token(0)
        gene_indices: list[int] = []
        counts: list[int] = []
        for i, tid in enumerate(count_tokens[: self.max_gene_count]):
            if tid == zero_id:
                continue
            count = self._decode_count_token(tid)
            if count > 0:
                gene_indices.append(i)
                counts.append(count)
        X = scipy.sparse.csr_matrix(
            (counts, ([0] * len(counts), gene_indices)),
            shape=(1, self.max_gene_count),
            dtype=np.int32,
        )
        return anndata.AnnData(
            X=X,
            obs=pd.DataFrame(obs, index=[0]),
            var=pd.DataFrame(index=pd.Index(list(self.gene_names_ordered))),
        )

    def encode_dense_counts(self, row: Any, gene_names: tuple[str, ...] | list[str]) -> list[int]:
        """Encode a gene-expression row into a dense list of count tokens.

        Maps each gene in *gene_names* to its dense index and writes the
        corresponding count token at that position.  Genes not in the panel
        (``gene_name_to_dense_idx``) are silently skipped.  All positions
        default to the zero-count token.

        Handles both sparse (``scipy.sparse``) and dense array inputs.

        Args:
            row: Expression values for a single cell.  Accepted types:

                - ``scipy.sparse`` matrix (any format; converted to CSR).
                - ``numpy.ndarray`` or array-like (1-D after flattening).

            gene_names: Gene names aligned with the columns of *row*.
                Length must match the number of columns in *row*.

        Returns:
            List of ``max_gene_count`` integer token ids.  Position ``i``
            holds the count token for ``gene_names_ordered[i]``.
        """
        zero_id = self._encode_count_token(0)
        tokens = [zero_id] * self.max_gene_count

        if scipy.sparse.issparse(row):
            csr = row.tocsr()
            for local_idx, count in zip(csr.indices.tolist(), csr.data.tolist()):
                gene_name = str(gene_names[local_idx])
                dense_idx = self.gene_name_to_dense_idx.get(gene_name)
                if dense_idx is not None:
                    tokens[dense_idx] = self._encode_count_token(int(count))
            return tokens

        row_array = np.asarray(row).flatten()
        for gene_idx, gene_name in enumerate(gene_names):
            dense_idx = self.gene_name_to_dense_idx.get(gene_name)
            if dense_idx is None:
                continue
            count_val = int(row_array[gene_idx]) if gene_idx < len(row_array) else 0
            if count_val > 0:
                tokens[dense_idx] = self._encode_count_token(count_val)
        return tokens
