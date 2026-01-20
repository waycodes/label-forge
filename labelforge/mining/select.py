"""
Hard-negative selection.

Combines hardness scores and candidate pairs to select final hard negatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from labelforge.mining.candidates import CandidatePair
from labelforge.mining.hardness import HardnessResult


class SelectionMode(str, Enum):
    """Mode for hard negative selection."""

    PER_ANCHOR = "per_anchor"  # Select top-K per anchor
    GLOBAL = "global"  # Select top-K overall
    STRATIFIED = "stratified"  # Balanced selection across classes


@dataclass
class HardNegative:
    """A selected hard negative example."""

    anchor_row_id: str
    negative_row_id: str
    hardness_score: float
    similarity_score: float
    combined_score: float
    selection_rank: int = 0
    justification: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anchor_row_id": self.anchor_row_id,
            "negative_row_id": self.negative_row_id,
            "hardness_score": self.hardness_score,
            "similarity_score": self.similarity_score,
            "combined_score": self.combined_score,
            "selection_rank": self.selection_rank,
            "justification": self.justification,
            "metadata": self.metadata,
        }


@dataclass
class SelectionConfig:
    """Configuration for hard negative selection."""

    mode: SelectionMode = SelectionMode.PER_ANCHOR
    top_k_per_anchor: int = 5
    top_k_global: int = 1000
    min_similarity: float = 0.5
    max_similarity: float = 0.99
    min_hardness: float = 0.3
    similarity_weight: float = 0.5
    hardness_weight: float = 0.5


def compute_combined_score(
    similarity: float,
    hardness: float,
    config: SelectionConfig,
) -> float:
    """
    Compute combined selection score.

    Args:
        similarity: Embedding similarity score.
        hardness: Hardness score.
        config: Selection configuration.

    Returns:
        Combined score in [0, 1].
    """
    total_weight = config.similarity_weight + config.hardness_weight
    if total_weight == 0:
        return 0.0

    return (
        config.similarity_weight * similarity + config.hardness_weight * hardness
    ) / total_weight


def generate_justification(
    similarity: float,
    hardness: float,
    hardness_components: dict[str, float],
) -> str:
    """
    Generate human-readable justification for selection.

    Args:
        similarity: Embedding similarity.
        hardness: Overall hardness score.
        hardness_components: Individual hardness metrics.

    Returns:
        Justification string.
    """
    parts = []

    if similarity > 0.9:
        parts.append(f"very high similarity ({similarity:.2f})")
    elif similarity > 0.7:
        parts.append(f"high similarity ({similarity:.2f})")
    else:
        parts.append(f"moderate similarity ({similarity:.2f})")

    if "score" in hardness_components:
        score_hardness = hardness_components["score"]
        if score_hardness > 0.7:
            parts.append("low quality score")
        elif score_hardness > 0.4:
            parts.append("medium quality score")

    if "uncertainty" in hardness_components:
        unc = hardness_components["uncertainty"]
        if unc > 0.7:
            parts.append("high model uncertainty")

    return "; ".join(parts) if parts else "selected by combined criteria"


def select_hard_negatives(
    candidates: list[CandidatePair],
    hardness_results: dict[str, HardnessResult],
    config: SelectionConfig,
) -> list[HardNegative]:
    """
    Select hard negatives from candidates and hardness scores.

    Args:
        candidates: Candidate pairs from generation.
        hardness_results: Hardness results keyed by row_id.
        config: Selection configuration.

    Returns:
        List of selected hard negatives.
    """
    selections: list[HardNegative] = []

    for cand in candidates:
        # Get hardness for the candidate (the potential negative)
        hardness = hardness_results.get(cand.candidate_row_id)
        hardness_score = hardness.hardness_score if hardness else 0.0
        hardness_components = hardness.components if hardness else {}

        # Filter by thresholds
        if cand.similarity < config.min_similarity:
            continue
        if cand.similarity > config.max_similarity:
            continue
        if hardness_score < config.min_hardness:
            continue

        # Compute combined score
        combined = compute_combined_score(
            similarity=cand.similarity,
            hardness=hardness_score,
            config=config,
        )

        # Generate justification
        justification = generate_justification(
            similarity=cand.similarity,
            hardness=hardness_score,
            hardness_components=hardness_components,
        )

        selections.append(
            HardNegative(
                anchor_row_id=cand.anchor_row_id,
                negative_row_id=cand.candidate_row_id,
                hardness_score=hardness_score,
                similarity_score=cand.similarity,
                combined_score=combined,
                justification=justification,
                metadata=cand.metadata,
            )
        )

    # Apply selection mode
    if config.mode == SelectionMode.PER_ANCHOR:
        selections = _select_per_anchor(selections, config.top_k_per_anchor)
    elif config.mode == SelectionMode.GLOBAL:
        selections = _select_global(selections, config.top_k_global)
    else:  # STRATIFIED
        selections = _select_global(selections, config.top_k_global)

    return selections


def _select_per_anchor(
    selections: list[HardNegative],
    top_k: int,
) -> list[HardNegative]:
    """Select top-K per anchor."""
    # Group by anchor
    by_anchor: dict[str, list[HardNegative]] = {}
    for sel in selections:
        if sel.anchor_row_id not in by_anchor:
            by_anchor[sel.anchor_row_id] = []
        by_anchor[sel.anchor_row_id].append(sel)

    # Select top-K per anchor
    result: list[HardNegative] = []
    for anchor_id, anchor_sels in sorted(by_anchor.items()):
        # Sort by combined score
        anchor_sels.sort(
            key=lambda s: (-s.combined_score, s.negative_row_id),
        )
        top = anchor_sels[:top_k]

        for rank, sel in enumerate(top, 1):
            sel.selection_rank = rank

        result.extend(top)

    return result


def _select_global(
    selections: list[HardNegative],
    top_k: int,
) -> list[HardNegative]:
    """Select top-K globally."""
    # Sort by combined score
    selections.sort(
        key=lambda s: (-s.combined_score, s.anchor_row_id, s.negative_row_id),
    )

    top = selections[:top_k]

    for rank, sel in enumerate(top, 1):
        sel.selection_rank = rank

    return top


def format_selection_report(
    selections: list[HardNegative],
    include_metadata: bool = False,
) -> str:
    """
    Format selection results as a report.

    Args:
        selections: Selected hard negatives.
        include_metadata: Whether to include metadata.

    Returns:
        Formatted report string.
    """
    lines = [
        f"Hard Negative Selection Report",
        f"=" * 40,
        f"Total selected: {len(selections)}",
        "",
    ]

    # Group by anchor
    by_anchor: dict[str, list[HardNegative]] = {}
    for sel in selections:
        if sel.anchor_row_id not in by_anchor:
            by_anchor[sel.anchor_row_id] = []
        by_anchor[sel.anchor_row_id].append(sel)

    lines.append(f"Anchors with negatives: {len(by_anchor)}")
    lines.append("")

    # Sample entries
    for anchor_id, anchor_sels in list(by_anchor.items())[:5]:
        lines.append(f"Anchor: {anchor_id}")
        for sel in anchor_sels[:3]:
            lines.append(
                f"  → {sel.negative_row_id} "
                f"(combined={sel.combined_score:.3f}, "
                f"sim={sel.similarity_score:.3f})"
            )
        if len(anchor_sels) > 3:
            lines.append(f"  ... and {len(anchor_sels) - 3} more")
        lines.append("")

    if len(by_anchor) > 5:
        lines.append(f"... and {len(by_anchor) - 5} more anchors")

    return "\n".join(lines)
