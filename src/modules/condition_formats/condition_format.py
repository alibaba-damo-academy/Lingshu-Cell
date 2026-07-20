"""Declarative condition-prefix format schema.

Describes the slot layout of a condition prefix without encoding logic.
Encoding is owned by the tokenizer; this module only describes structure.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class ConditionSlot:
    """One semantic slot inside the condition prefix.

    Attributes:
        name: Unique identifier for this slot (e.g. ``"cell_type"``,
            ``"target_gene"``).  Used as the key in prefix-position maps.
        slot_type: Encoding strategy.  Supported values:

            - ``"categorical"``: looked up in a vocab codebook dict.
            - ``"gene_index"``: encoded as ``offset + dense_idx``.
            - ``"special_token"``: looked up directly in special_token_vocab.

        width: Number of token positions this slot occupies.
            Only ``width=1`` is currently supported by the tokenizer.
        obs_key: Primary column name in ``adata.obs`` that supplies the
            raw value for this slot at encode time.
        override_value: When non-empty, this value is forced for **all**
            samples regardless of what the data provides.  Useful for
            unconditional generation (e.g. override all conditions to a
            fixed label).  When empty (the default), the slot value must
            come from the data (``obs_key`` or ``alias_obs_key``); a
            missing value raises ``ValueError`` at encode / normalize
            time.
        codebook_key: Key into the vocab JSON that holds the id-mapping
            dict for this slot (e.g. ``"cell_type_token_vocab"``).
            Empty string means no codebook.
        alias_obs_key: Optional alternate column name in ``adata.obs``.
            Checked when *obs_key* is missing (e.g. ``"gene_target"``
            as an alias for ``"target_gene"``).
        default_value: Fallback value used when the obs value for a
            specific sample is empty or NaN.  Unlike ``override_value``
            (which forces ALL samples), ``default_value`` only applies
            to individual empty cells.  Empty string means no fallback.
    """

    name: str
    slot_type: str
    width: int
    obs_key: str
    override_value: str = ""
    codebook_key: str = ""
    alias_obs_key: str = ""
    default_value: str = ""

    def resolve_value(self, obs_row: Mapping[str, Any]) -> str:
        """Resolve the value for this slot from an observation row.

        When ``override_value`` is set (non-empty), it is returned
        unconditionally.  Otherwise the lookup chain is
        ``obs_key`` → ``alias_obs_key`` → ``ValueError``.

        Args:
            obs_row: A mapping-like object (e.g. ``pandas.Series``, dict)
                containing observation metadata.

        Returns:
            Resolved string value for this slot.

        Raises:
            ValueError: If no value can be resolved and ``override_value``
                is empty.
        """
        if self.override_value:
            return self.override_value
        value = obs_row.get(self.obs_key)
        if value is None or str(value) == "":
            if self.alias_obs_key:
                value = obs_row.get(self.alias_obs_key)
            if value is None or str(value) == "":
                if self.default_value:
                    return self.default_value
                raise ValueError(
                    f"Slot '{self.name}': no value found via "
                    f"obs_key='{self.obs_key}'"
                    f"{f' or alias={self.alias_obs_key!r}' if self.alias_obs_key else ''}"
                    f" and no override_value/default_value is set."
                )
        return str(value)


@dataclass(frozen=True)
class ControlIndicator:
    """Identifies which obs value marks a sample as control / non-targeting.

    Attributes:
        obs_key: Column name in ``adata.obs`` that carries the control
            indicator (e.g. ``"treatment"`` or ``"target_gene"``).
        control_value: The string value that marks a sample as control
            (e.g. ``"DMSO"``, ``"untreated"``, or ``"non-targeting"``).
            Comparisons are case-insensitive.
    """

    obs_key: str
    control_value: str


@dataclass(frozen=True)
class ConditionFormat:
    """Full description of a condition prefix layout.

    The prefix is a fixed-length token sequence placed before the gene
    expression tokens.  It encodes experimental conditions such as cell
    type and perturbation target.

    ``bookend_positions`` are absolute indices within the prefix that
    receive the ``bookend_token`` (e.g. ``<cond>``).  Semantic slots
    fill the remaining positions in declaration order.

    Example VCC layout (prefix_length=5)::

        pos 0: <cond>          (bookend)
        pos 1: cell_type       (categorical slot)
        pos 2: target_gene     (gene_index slot)
        pos 3: <cond>          (bookend)
        pos 4: <cond>          (bookend)

    Example drug-screen layout (prefix_length=3)::

        pos 0: <cond>          (bookend)
        pos 1: cell_line       (categorical slot)
        pos 2: treatment       (categorical slot)

    Attributes:
        slots: Ordered tuple of semantic slots.
        bookend_token: Special token name used at bookend positions
            (must exist in ``special_token_vocab``).
        bookend_positions: Absolute prefix indices occupied by
            *bookend_token*.
        control_indicator: Optional descriptor for identifying
            control/non-targeting samples in the dataset.
    """

    slots: tuple[ConditionSlot, ...] = field(default_factory=tuple)
    bookend_token: str = "<cond>"
    bookend_positions: tuple[int, ...] = field(default_factory=tuple)
    control_indicator: ControlIndicator | None = None

    def __post_init__(self) -> None:
        """Convert OmegaConf DictConfigs to real Python instances.

        Hydra's ``instantiate()`` may leave nested list elements as
        OmegaConf DictConfigs rather than actual dataclass instances.
        This prevents method calls like ``slot.resolve_value()``.
        """
        real_slots = []
        for s in self.slots:
            if isinstance(s, ConditionSlot):
                real_slots.append(s)
            else:
                d = {k: v for k, v in dict(s).items() if not k.startswith("_")}
                real_slots.append(ConditionSlot(**d))
        object.__setattr__(self, "slots", tuple(real_slots))

        if not isinstance(self.bookend_positions, tuple):
            object.__setattr__(self, "bookend_positions", tuple(self.bookend_positions))

        ci = self.control_indicator
        if ci is not None and not isinstance(ci, ControlIndicator):
            d = {k: v for k, v in dict(ci).items() if not k.startswith("_")}
            object.__setattr__(self, "control_indicator", ControlIndicator(**d))

    @property
    def prefix_length(self) -> int:
        """Total number of token positions in the condition prefix."""
        return sum(s.width for s in self.slots) + len(self.bookend_positions)

    def obs_keys(self) -> tuple[str, ...]:
        """Return the primary obs column names required by all slots."""
        return tuple(s.obs_key for s in self.slots)

    def slot_by_name(self, name: str) -> ConditionSlot | None:
        """Look up a slot by its unique name.

        Args:
            name: Slot name to search for.

        Returns:
            The matching ``ConditionSlot``, or ``None`` if not found.
        """
        for s in self.slots:
            if s.name == name:
                return s
        return None

    # ------------------------------------------------------------------
    # Obs-level behaviour
    # ------------------------------------------------------------------

    def resolve_obs_values(self, obs_row: Mapping[str, Any]) -> dict[str, str]:
        """Resolve condition values for all slots from an observation row.

        Delegates to each slot's ``resolve_value`` method which returns
        the override when set, or applies the lookup chain
        ``obs_key`` → ``alias_obs_key`` → ``ValueError``.

        Args:
            obs_row: A mapping-like object (e.g. ``pandas.Series``, dict)
                containing observation metadata.

        Returns:
            Dict mapping each slot's ``obs_key`` to its resolved string value.
        """
        return {slot.obs_key: slot.resolve_value(obs_row) for slot in self.slots}

    def normalize_obs_dataframe(
        self,
        obs: pd.DataFrame,
        *,
        canonical_control_label: str = "",
    ) -> pd.DataFrame:
        """Normalize condition-relevant columns in an obs DataFrame.

        Performs two operations in order:

        1. **Slot column resolution** — for each slot declared in
           ``self.slots``:

           - If ``override_value`` is non-empty, the entire column is
             set to that value regardless of existing data.
           - Otherwise, if *obs_key* is missing but *alias_obs_key* is
             present, copy the alias column to *obs_key*.
           - If neither is present and no override, raise ``ValueError``.
           - If both primary and alias are present but differ, the alias
             column wins and a ``UserWarning`` is emitted.

        2. **Control value normalization** — if ``control_indicator`` is
           configured and *canonical_control_label* is non-empty, performs
           a case-insensitive string replacement of
           ``control_indicator.control_value`` with *canonical_control_label*
           in the relevant obs column.

        Args:
            obs: Observation metadata ``DataFrame``.  A copy is made
                internally; the caller's frame is not mutated.
            canonical_control_label: The canonical control label to
                substitute (e.g. ``"non-targeting"``).  Pass an empty
                string to skip control normalization.

        Returns:
            A new ``DataFrame`` with condition columns resolved and
            control values normalized.
        """
        obs = obs.copy()

        # Resolve condition-slot columns from override, primary key, or alias
        for slot in self.slots:
            if slot.override_value:
                obs[slot.obs_key] = slot.override_value
            elif slot.obs_key not in obs.columns:
                if slot.alias_obs_key and slot.alias_obs_key in obs.columns:
                    obs[slot.obs_key] = obs[slot.alias_obs_key]
                elif slot.default_value:
                    obs[slot.obs_key] = slot.default_value
                else:
                    raise ValueError(
                        f"Slot '{slot.name}': obs_key '{slot.obs_key}' not "
                        f"found in DataFrame columns"
                        f"{f' (alias {slot.alias_obs_key!r} also missing)' if slot.alias_obs_key else ''}"
                        f" and no override_value/default_value is set."
                    )
            elif slot.alias_obs_key and slot.alias_obs_key in obs.columns:
                alias_col = obs[slot.alias_obs_key]
                if not obs[slot.obs_key].equals(alias_col):
                    warnings.warn(
                        f"{slot.obs_key} and {slot.alias_obs_key} "
                        f"columns differ. Using {slot.alias_obs_key}.",
                        stacklevel=2,
                    )
                obs[slot.obs_key] = alias_col

            # Fill empty/NaN values with default_value (per-cell fallback)
            if slot.default_value and slot.obs_key in obs.columns:
                col = obs[slot.obs_key].astype(str)
                empty_mask = (col == "") | (col == "nan") | col.isna()
                if empty_mask.any():
                    series = obs[slot.obs_key]
                    if hasattr(series, "cat"):
                        series = series.cat.add_categories(
                            [c for c in [slot.default_value] if c not in series.cat.categories]
                        )
                        obs[slot.obs_key] = series
                    obs.loc[empty_mask, slot.obs_key] = slot.default_value

        # Normalize control-indicator values to canonical label
        if canonical_control_label and self.control_indicator is not None:
            ci = self.control_indicator
            if ci.obs_key in obs.columns:
                obs[ci.obs_key] = (
                    obs[ci.obs_key]
                    .astype(str)
                    .str.replace(
                        ci.control_value,
                        canonical_control_label,
                        case=False,
                        regex=False,
                    )
                )

        return obs

    @property
    def gene_index_slot(self) -> ConditionSlot | None:
        """Return the first slot with ``slot_type == "gene_index"``, or ``None``.

        Useful for locating the gene perturbation slot without hardcoding
        a slot name.
        """
        for s in self.slots:
            if s.slot_type == "gene_index":
                return s
        return None

    # ------------------------------------------------------------------
    # Prefix layout behaviour
    # ------------------------------------------------------------------

    @property
    def slot_position_map(self) -> dict[str, int]:
        """Map each slot name to its absolute position in the condition prefix.

        Iterates positions ``0 .. prefix_length - 1`` in order, skipping
        bookend positions, and assigns each remaining position to the next
        slot in declaration order.

        Returns:
            Dict mapping slot name (e.g. ``"cell_type"``) to its zero-based
            index within the condition prefix.
        """
        bookend_set = set(self.bookend_positions)
        position_map: dict[str, int] = {}
        slot_iter = iter(self.slots)
        for pos in range(self.prefix_length):
            if pos in bookend_set:
                continue
            slot = next(slot_iter, None)
            if slot is not None:
                position_map[slot.name] = pos
        return position_map

    def build_prefix(
        self,
        conditions: Mapping[str, str],
        bookend_token_id: int,
        encode_slot: Callable[["ConditionSlot", str], int],
    ) -> list[int]:
        """Build the condition prefix token list.

        Fills bookend positions with *bookend_token_id*, then encodes each
        slot value using the provided callback.  The callback receives a
        ``ConditionSlot`` and the resolved string value, and must return
        the integer token ID for that slot position.

        Args:
            conditions: Resolved condition values keyed by slot ``obs_key``.
            bookend_token_id: Token ID to place at bookend positions.
            encode_slot: Callback ``(slot, value) -> token_id`` that
                encodes one slot's string value to its token ID.

        Returns:
            List of ``prefix_length`` integer token IDs.
        """
        bookend_set = set(self.bookend_positions)
        prefix: list[int] = [0] * self.prefix_length
        for pos in bookend_set:
            prefix[pos] = bookend_token_id
        slot_iter = iter(self.slots)
        for pos in range(self.prefix_length):
            if pos in bookend_set:
                continue
            slot = next(slot_iter, None)
            if slot is None:
                break
            value = conditions.get(slot.obs_key, slot.override_value)
            prefix[pos] = encode_slot(slot, str(value))
        return prefix

    def decode_prefix(
        self,
        token_ids: Sequence[int],
        decode_slot: Callable[["ConditionSlot", int], str],
    ) -> dict[str, str]:
        """Decode condition values from prefix token IDs.

        For each slot, looks up its position in the prefix via
        ``slot_position_map`` and decodes the token ID at that position
        using the provided callback.  Falls back to ``slot.override_value``
        when the position is out of range.

        Args:
            token_ids: Token ID sequence (at least ``prefix_length`` long).
            decode_slot: Callback ``(slot, token_id) -> value_str`` that
                decodes one token ID to a slot's string value.

        Returns:
            Dict mapping each slot's ``obs_key`` to its decoded string value.
        """
        pos_map = self.slot_position_map
        result: dict[str, str] = {}
        for slot in self.slots:
            # Skip slots without an obs column (e.g. ESM/special-token placeholders);
            # their values are internal to the prefix encoding only.
            if not slot.obs_key:
                continue
            pos = pos_map.get(slot.name)
            if pos is not None and pos < len(token_ids):
                result[slot.obs_key] = decode_slot(slot, int(token_ids[pos]))
            else:
                result[slot.obs_key] = slot.override_value
        return result
