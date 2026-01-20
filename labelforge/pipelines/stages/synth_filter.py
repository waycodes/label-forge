"""
Synthetic Data Filtering Stage.

Filters synthetic data based on quality scores and diversity constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from labelforge.pipelines.stage import Stage, StageConfig, StageContext, StageType

if TYPE_CHECKING:
    import ray.data


class SynthFilterStage(Stage):
    """
    Stage that filters synthetic data based on quality criteria.

    Supports:
    - Score-based filtering (minimum quality threshold)
    - Content-based filtering (length, patterns)
    - Diversity constraints

    Required inputs:
    - row_id: Unique row identifier
    - score field (configurable)

    Outputs:
    - filtered: Boolean indicating if filtered out
    - filter_reason: Reason for filtering (if filtered)
    """

    def __init__(
        self,
        config: StageConfig,
        *,
        score_field: str = "quality_score",
        min_score: float = 5.0,
        content_field: str | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        banned_patterns: list[str] | None = None,
    ):
        """
        Initialize synthetic data filter stage.

        Args:
            config: Stage configuration.
            score_field: Field containing quality score.
            min_score: Minimum score threshold.
            content_field: Optional field to check for content filters.
            min_length: Minimum content length.
            max_length: Maximum content length.
            banned_patterns: Patterns that trigger filtering.
        """
        super().__init__(config)
        self.score_field = score_field
        self.min_score = min_score
        self.content_field = content_field
        self.min_length = min_length
        self.max_length = max_length
        self.banned_patterns = banned_patterns or []

    @property
    def input_schema(self) -> dict[str, type]:
        """Required input fields."""
        schema = {"row_id": str}
        if self.score_field:
            schema[self.score_field] = float
        return schema

    @property
    def output_schema(self) -> dict[str, type]:
        """Output fields added by this stage."""
        return {
            "filtered": bool,
            "filter_reason": str,
        }

    def _check_filters(self, row: dict[str, Any]) -> tuple[bool, str]:
        """
        Check if row should be filtered.

        Returns:
            Tuple of (should_filter, reason).
        """
        # Check score threshold
        if self.score_field and self.score_field in row:
            score = row[self.score_field]
            if score is not None and score < self.min_score:
                return True, f"score {score:.2f} below threshold {self.min_score}"

        # Check content length
        if self.content_field and self.content_field in row:
            content = str(row[self.content_field])

            if self.min_length and len(content) < self.min_length:
                return True, f"content too short ({len(content)} < {self.min_length})"

            if self.max_length and len(content) > self.max_length:
                return True, f"content too long ({len(content)} > {self.max_length})"

            # Check banned patterns
            content_lower = content.lower()
            for pattern in self.banned_patterns:
                if pattern.lower() in content_lower:
                    return True, f"contains banned pattern: {pattern}"

        return False, ""

    def run(
        self,
        dataset: ray.data.Dataset,
        context: StageContext,
    ) -> ray.data.Dataset:
        """
        Execute the filter stage on a dataset.

        Args:
            dataset: Input Ray Dataset.
            context: Execution context.

        Returns:
            Dataset with filter annotations (not removed).
        """
        check_filters = self._check_filters

        def apply_filter(row: dict[str, Any]) -> dict[str, Any]:
            """Apply filters to a row."""
            filtered, reason = check_filters(row)

            result = dict(row)
            result["filtered"] = filtered
            result["filter_reason"] = reason

            return result

        return dataset.map(apply_filter)

    def run_remove(
        self,
        dataset: ray.data.Dataset,
        context: StageContext,
    ) -> ray.data.Dataset:
        """
        Execute filter and remove filtered rows.

        Args:
            dataset: Input Ray Dataset.
            context: Execution context.

        Returns:
            Dataset with filtered rows removed.
        """
        check_filters = self._check_filters

        def should_keep(row: dict[str, Any]) -> bool:
            """Check if row should be kept."""
            filtered, _ = check_filters(row)
            return not filtered

        return dataset.filter(should_keep)


def create_synth_filter_stage(
    name: str = "synth_filter",
    *,
    score_field: str = "quality_score",
    min_score: float = 5.0,
    content_field: str | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    banned_patterns: list[str] | None = None,
    version: str = "1.0.0",
) -> SynthFilterStage:
    """
    Create a synthetic data filter stage.

    Args:
        name: Stage name.
        score_field: Field with quality score.
        min_score: Minimum score threshold.
        content_field: Field to check content.
        min_length: Minimum content length.
        max_length: Maximum content length.
        banned_patterns: Banned content patterns.
        version: Stage version.

    Returns:
        Configured SynthFilterStage.
    """
    config = StageConfig(
        name=name,
        stage_type=StageType.FILTER,
        version=version,
        model_spec_name=None,
        params={
            "score_field": score_field,
            "min_score": min_score,
            "min_length": min_length,
            "max_length": max_length,
        },
    )

    return SynthFilterStage(
        config=config,
        score_field=score_field,
        min_score=min_score,
        content_field=content_field,
        min_length=min_length,
        max_length=max_length,
        banned_patterns=banned_patterns,
    )
