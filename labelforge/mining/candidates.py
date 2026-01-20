"""
Candidate generation for hard-negative mining.

Generates candidate pairs based on embedding similarity and other criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CandidatePair:
    """A candidate pair for hard-negative mining."""

    anchor_row_id: str
    candidate_row_id: str
    similarity: float
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anchor_row_id": self.anchor_row_id,
            "candidate_row_id": self.candidate_row_id,
            "similarity": self.similarity,
            "rank": self.rank,
            "metadata": self.metadata,
        }


@dataclass
class CandidateGenerationConfig:
    """Configuration for candidate generation."""

    top_k: int = 100
    min_similarity: float = 0.5
    max_similarity: float = 1.0
    exclude_same_class: bool = True
    class_field: str = "label"
    embedding_field: str = "embedding"
    batch_size: int = 1000


def generate_candidates_from_embeddings(
    anchor_id: str,
    anchor_embedding: np.ndarray,
    anchor_class: str | None,
    candidate_ids: list[str],
    candidate_embeddings: np.ndarray,
    candidate_classes: list[str | None],
    config: CandidateGenerationConfig,
) -> list[CandidatePair]:
    """
    Generate candidate pairs based on embedding similarity.

    Args:
        anchor_id: ID of anchor row.
        anchor_embedding: Embedding of anchor.
        anchor_class: Class label of anchor (for filtering).
        candidate_ids: IDs of candidate rows.
        candidate_embeddings: Embeddings of candidates (2D array).
        candidate_classes: Class labels of candidates.
        config: Generation configuration.

    Returns:
        List of CandidatePair sorted by similarity.
    """
    from labelforge.mining.hardness import batch_cosine_similarity

    # Compute similarities
    similarities = batch_cosine_similarity(anchor_embedding, candidate_embeddings)

    # Build candidate list
    candidates: list[CandidatePair] = []

    for i, (cand_id, sim, cand_class) in enumerate(
        zip(candidate_ids, similarities, candidate_classes)
    ):
        # Skip self
        if cand_id == anchor_id:
            continue

        # Filter by similarity range
        if sim < config.min_similarity or sim > config.max_similarity:
            continue

        # Optionally exclude same class
        if config.exclude_same_class and anchor_class and cand_class:
            if anchor_class == cand_class:
                continue

        candidates.append(
            CandidatePair(
                anchor_row_id=anchor_id,
                candidate_row_id=cand_id,
                similarity=float(sim),
                metadata={"candidate_class": cand_class},
            )
        )

    # Sort by similarity (descending) with stable tie-breaking by ID
    candidates.sort(key=lambda c: (-c.similarity, c.candidate_row_id))

    # Take top-k and assign ranks
    top_candidates = candidates[: config.top_k]
    for rank, cand in enumerate(top_candidates, 1):
        cand.rank = rank

    return top_candidates


def generate_candidates_batch(
    rows: list[dict[str, Any]],
    config: CandidateGenerationConfig,
) -> dict[str, list[CandidatePair]]:
    """
    Generate candidates for all rows in a batch.

    Args:
        rows: List of row dictionaries with embeddings.
        config: Generation configuration.

    Returns:
        Dict mapping anchor row_id to list of candidates.
    """
    # Extract embeddings and metadata
    row_ids = [r["row_id"] for r in rows]
    embeddings = np.array([r[config.embedding_field] for r in rows])
    classes = [r.get(config.class_field) for r in rows]

    results: dict[str, list[CandidatePair]] = {}

    for i, (anchor_id, anchor_emb, anchor_class) in enumerate(
        zip(row_ids, embeddings, classes)
    ):
        candidates = generate_candidates_from_embeddings(
            anchor_id=anchor_id,
            anchor_embedding=anchor_emb,
            anchor_class=anchor_class,
            candidate_ids=row_ids,
            candidate_embeddings=embeddings,
            candidate_classes=classes,
            config=config,
        )
        results[anchor_id] = candidates

    return results


def filter_candidates_by_class(
    candidates: list[CandidatePair],
    class_mapping: dict[str, str],
    exclude_same: bool = True,
    target_classes: list[str] | None = None,
) -> list[CandidatePair]:
    """
    Filter candidates by class criteria.

    Args:
        candidates: Input candidate pairs.
        class_mapping: Mapping from row_id to class label.
        exclude_same: Exclude same-class pairs.
        target_classes: Only include candidates from these classes.

    Returns:
        Filtered candidates.
    """
    filtered: list[CandidatePair] = []

    for cand in candidates:
        anchor_class = class_mapping.get(cand.anchor_row_id)
        cand_class = class_mapping.get(cand.candidate_row_id)

        if exclude_same and anchor_class and cand_class:
            if anchor_class == cand_class:
                continue

        if target_classes and cand_class not in target_classes:
            continue

        filtered.append(cand)

    return filtered


def merge_candidate_lists(
    lists: list[list[CandidatePair]],
    top_k: int | None = None,
    dedupe: bool = True,
) -> list[CandidatePair]:
    """
    Merge multiple candidate lists.

    Args:
        lists: Lists of candidates to merge.
        top_k: Optional limit on merged results.
        dedupe: Remove duplicate pairs.

    Returns:
        Merged and sorted candidate list.
    """
    merged: list[CandidatePair] = []

    for lst in lists:
        merged.extend(lst)

    if dedupe:
        seen = set()
        deduped = []
        for cand in merged:
            key = (cand.anchor_row_id, cand.candidate_row_id)
            if key not in seen:
                seen.add(key)
                deduped.append(cand)
        merged = deduped

    # Sort by similarity
    merged.sort(key=lambda c: (-c.similarity, c.anchor_row_id, c.candidate_row_id))

    # Reassign ranks
    for rank, cand in enumerate(merged, 1):
        cand.rank = rank

    if top_k:
        return merged[:top_k]

    return merged


class CandidateIndex:
    """
    Index for efficient candidate retrieval.

    Uses approximate nearest neighbor search for large datasets.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        row_ids: list[str],
        classes: list[str | None] | None = None,
    ):
        """
        Initialize candidate index.

        Args:
            embeddings: Array of embeddings (2D).
            row_ids: List of row IDs.
            classes: Optional list of class labels.
        """
        self.embeddings = embeddings
        self.row_ids = row_ids
        self.classes = classes or [None] * len(row_ids)
        self._id_to_idx = {rid: i for i, rid in enumerate(row_ids)}

    def query(
        self,
        anchor_id: str,
        config: CandidateGenerationConfig,
    ) -> list[CandidatePair]:
        """
        Query for candidates of an anchor.

        Args:
            anchor_id: Anchor row ID.
            config: Configuration.

        Returns:
            List of candidate pairs.
        """
        if anchor_id not in self._id_to_idx:
            return []

        idx = self._id_to_idx[anchor_id]
        anchor_emb = self.embeddings[idx]
        anchor_class = self.classes[idx]

        return generate_candidates_from_embeddings(
            anchor_id=anchor_id,
            anchor_embedding=anchor_emb,
            anchor_class=anchor_class,
            candidate_ids=self.row_ids,
            candidate_embeddings=self.embeddings,
            candidate_classes=self.classes,
            config=config,
        )

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        embedding_field: str = "embedding",
        class_field: str = "label",
    ) -> CandidateIndex:
        """
        Create index from rows.

        Args:
            rows: List of row dictionaries.
            embedding_field: Field containing embeddings.
            class_field: Field containing class labels.

        Returns:
            CandidateIndex.
        """
        row_ids = [r["row_id"] for r in rows]
        embeddings = np.array([r[embedding_field] for r in rows])
        classes = [r.get(class_field) for r in rows]

        return cls(embeddings=embeddings, row_ids=row_ids, classes=classes)
