"""
Hardness functions for mining hard negatives.

Computes hardness scores from multiple signals: rubric scores,
embedding similarity, and model uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class HardnessMetric(str, Enum):
    """Types of hardness metrics."""

    LOW_SCORE = "low_score"  # Low rubric/quality score
    HIGH_SIMILARITY = "high_similarity"  # Similar but different class
    UNCERTAINTY = "uncertainty"  # Model uncertainty
    NEAR_BOUNDARY = "near_boundary"  # Close to decision boundary
    COMPOSITE = "composite"  # Weighted combination


@dataclass
class HardnessConfig:
    """Configuration for hardness computation."""

    # Score-based hardness
    score_field: str = "score"
    score_threshold: float = 5.0  # Scores below this are "hard"
    invert_score: bool = True  # Lower score = higher hardness

    # Similarity-based hardness
    similarity_threshold: float = 0.9  # High similarity = hard
    embedding_field: str = "embedding"

    # Weights for composite metric
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "score": 0.4,
            "similarity": 0.4,
            "uncertainty": 0.2,
        }
    )

    # Selection
    top_k: int = 100
    min_hardness: float = 0.5


@dataclass
class HardnessResult:
    """Result of hardness computation for a single row."""

    row_id: str
    hardness_score: float
    metric: HardnessMetric
    components: dict[str, float] = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "row_id": self.row_id,
            "hardness_score": self.hardness_score,
            "metric": self.metric.value,
            "components": self.components,
            "rank": self.rank,
        }


def compute_score_hardness(
    score: float,
    min_score: float = 0.0,
    max_score: float = 10.0,
    invert: bool = True,
) -> float:
    """
    Compute hardness from a quality/rubric score.

    Args:
        score: Quality score.
        min_score: Minimum possible score.
        max_score: Maximum possible score.
        invert: If True, lower scores = higher hardness.

    Returns:
        Hardness score in [0, 1].
    """
    # Normalize to [0, 1]
    normalized = (score - min_score) / (max_score - min_score)
    normalized = max(0.0, min(1.0, normalized))

    if invert:
        return 1.0 - normalized
    return normalized


def compute_similarity_hardness(
    similarity: float,
    threshold: float = 0.9,
) -> float:
    """
    Compute hardness from embedding similarity.

    High similarity with different label = hard case.

    Args:
        similarity: Cosine similarity value.
        threshold: Threshold above which to consider hard.

    Returns:
        Hardness score in [0, 1].
    """
    if similarity >= threshold:
        # Scale from threshold to 1.0 -> hardness 0.5 to 1.0
        return 0.5 + 0.5 * (similarity - threshold) / (1.0 - threshold)
    else:
        # Scale from 0 to threshold -> hardness 0 to 0.5
        return 0.5 * similarity / threshold


def compute_uncertainty_hardness(
    confidence: float,
    entropy: float | None = None,
) -> float:
    """
    Compute hardness from model uncertainty.

    Low confidence or high entropy = hard case.

    Args:
        confidence: Model confidence (probability of top class).
        entropy: Optional prediction entropy.

    Returns:
        Hardness score in [0, 1].
    """
    # Lower confidence = higher hardness
    confidence_hardness = 1.0 - confidence

    if entropy is not None:
        # Normalize entropy (assuming max entropy is log(num_classes))
        # Higher entropy = more uncertain = harder
        entropy_hardness = min(1.0, entropy / 2.0)  # Rough normalization
        return 0.5 * confidence_hardness + 0.5 * entropy_hardness

    return confidence_hardness


def compute_composite_hardness(
    components: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute weighted composite hardness score.

    Args:
        components: Dict of metric name -> hardness value.
        weights: Optional weights (default: equal weighting).

    Returns:
        Composite hardness score in [0, 1].
    """
    if not components:
        return 0.0

    if weights is None:
        # Equal weighting
        weights = {k: 1.0 / len(components) for k in components}

    # Normalize weights
    total_weight = sum(weights.get(k, 0.0) for k in components)
    if total_weight == 0:
        return 0.0

    score = 0.0
    for metric, value in components.items():
        weight = weights.get(metric, 0.0)
        score += (weight / total_weight) * value

    return score


def compute_hardness(
    row: dict[str, Any],
    config: HardnessConfig,
    anchor_embedding: np.ndarray | None = None,
) -> HardnessResult:
    """
    Compute hardness for a single row.

    Args:
        row: Row data dictionary.
        config: Hardness configuration.
        anchor_embedding: Optional embedding for similarity computation.

    Returns:
        HardnessResult with scores.
    """
    row_id = row.get("row_id", "")
    components: dict[str, float] = {}

    # Score-based hardness
    if config.score_field in row:
        score = float(row[config.score_field])
        components["score"] = compute_score_hardness(
            score=score,
            invert=config.invert_score,
        )

    # Similarity-based hardness
    if config.embedding_field in row and anchor_embedding is not None:
        row_embedding = np.array(row[config.embedding_field])
        similarity = cosine_similarity(anchor_embedding, row_embedding)
        components["similarity"] = compute_similarity_hardness(
            similarity=similarity,
            threshold=config.similarity_threshold,
        )

    # Uncertainty-based hardness
    if "confidence" in row:
        confidence = float(row["confidence"])
        entropy = row.get("entropy")
        components["uncertainty"] = compute_uncertainty_hardness(
            confidence=confidence,
            entropy=float(entropy) if entropy else None,
        )

    # Compute composite
    if len(components) > 1:
        hardness_score = compute_composite_hardness(components, config.weights)
        metric = HardnessMetric.COMPOSITE
    elif "score" in components:
        hardness_score = components["score"]
        metric = HardnessMetric.LOW_SCORE
    elif "similarity" in components:
        hardness_score = components["similarity"]
        metric = HardnessMetric.HIGH_SIMILARITY
    elif "uncertainty" in components:
        hardness_score = components["uncertainty"]
        metric = HardnessMetric.UNCERTAINTY
    else:
        hardness_score = 0.0
        metric = HardnessMetric.COMPOSITE

    return HardnessResult(
        row_id=row_id,
        hardness_score=hardness_score,
        metric=metric,
        components=components,
    )


def rank_by_hardness(
    results: list[HardnessResult],
    top_k: int | None = None,
    min_hardness: float = 0.0,
) -> list[HardnessResult]:
    """
    Rank results by hardness score.

    Args:
        results: List of hardness results.
        top_k: Optional limit on results.
        min_hardness: Minimum hardness threshold.

    Returns:
        Sorted and filtered list of results.
    """
    # Filter by minimum hardness
    filtered = [r for r in results if r.hardness_score >= min_hardness]

    # Sort by hardness (descending)
    sorted_results = sorted(
        filtered,
        key=lambda r: (r.hardness_score, r.row_id),  # Stable sort
        reverse=True,
    )

    # Apply top-k
    if top_k:
        sorted_results = sorted_results[:top_k]

    # Assign ranks
    for i, result in enumerate(sorted_results):
        result.rank = i + 1

    return sorted_results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(
    anchor: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between anchor and multiple candidates.

    Args:
        anchor: Anchor embedding (1D).
        candidates: Candidate embeddings (2D: num_candidates x dim).

    Returns:
        Array of similarities.
    """
    anchor_norm = anchor / (np.linalg.norm(anchor) + 1e-8)
    candidates_norm = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
    return np.dot(candidates_norm, anchor_norm)
