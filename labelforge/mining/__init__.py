"""Hard-negative mining: hardness functions, candidate generation, selection."""

from labelforge.mining.candidates import (
    CandidateGenerationConfig,
    CandidateIndex,
    CandidatePair,
    generate_candidates_batch,
    generate_candidates_from_embeddings,
)
from labelforge.mining.hardness import (
    HardnessConfig,
    HardnessMetric,
    HardnessResult,
    compute_hardness,
    rank_by_hardness,
)
from labelforge.mining.select import (
    HardNegative,
    SelectionConfig,
    SelectionMode,
    select_hard_negatives,
)

__all__ = [
    # Hardness
    "HardnessConfig",
    "HardnessMetric",
    "HardnessResult",
    "compute_hardness",
    "rank_by_hardness",
    # Candidates
    "CandidateGenerationConfig",
    "CandidateIndex",
    "CandidatePair",
    "generate_candidates_batch",
    "generate_candidates_from_embeddings",
    # Selection
    "HardNegative",
    "SelectionConfig",
    "SelectionMode",
    "select_hard_negatives",
]
