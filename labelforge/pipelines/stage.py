"""
Pipeline stage abstraction.

A stage is a deterministic function from dataset to dataset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import xxhash

from labelforge.core.json_canonical import canonical_json_bytes

if TYPE_CHECKING:
    import ray.data


class StageType(str, Enum):
    """Type of pipeline stage."""

    VLM_CAPTION = "vlm_caption"
    VLM_TAGS = "vlm_tags"
    TEXT_LABEL = "text_label"
    TEXT_EXTRACT = "text_extract"
    RUBRIC_SCORE = "rubric_score"
    EMBED_TEXT = "embed_text"
    EMBED_MM = "embed_mm"
    RERANK = "rerank"
    SYNTH_TEXT = "synth_text"
    SYNTH_VLM = "synth_vlm"
    FILTER = "filter"
    TRANSFORM = "transform"


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""

    name: str
    stage_type: StageType
    version: str = "1.0.0"

    # Prompt configuration
    prompt_pack: str | None = None
    template_name: str | None = None
    rubric_name: str | None = None

    # Model configuration
    model_spec_name: str | None = None

    # Stage-specific config
    params: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Compute configuration fingerprint."""
        content = {
            "name": self.name,
            "stage_type": self.stage_type.value,
            "version": self.version,
            "prompt_pack": self.prompt_pack,
            "template_name": self.template_name,
            "rubric_name": self.rubric_name,
            "model_spec_name": self.model_spec_name,
            "params": self.params,
        }
        json_bytes = canonical_json_bytes(content)
        return xxhash.xxh64(json_bytes).hexdigest()


@dataclass
class StageResult:
    """Result of running a stage."""

    success: bool
    output_path: str | None = None
    output_row_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    latency_ms: float = 0.0
    error: str | None = None


class Stage(ABC):
    """
    Abstract base class for pipeline stages.

    A stage transforms an input dataset into an output dataset.
    """

    def __init__(self, config: StageConfig):
        """
        Initialize stage with configuration.

        Args:
            config: Stage configuration.
        """
        self.config = config
        self._created_at = datetime.utcnow()

    @property
    def name(self) -> str:
        """Stage name."""
        return self.config.name

    @property
    def version(self) -> str:
        """Stage version."""
        return self.config.version

    @property
    def stage_type(self) -> StageType:
        """Stage type."""
        return self.config.stage_type

    def fingerprint(self) -> str:
        """Compute stage fingerprint including config."""
        return self.config.fingerprint()

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, type]:
        """Required fields in input dataset."""
        ...

    @property
    @abstractmethod
    def output_schema(self) -> dict[str, type]:
        """Fields added to output dataset."""
        ...

    @abstractmethod
    def run(
        self,
        dataset: ray.data.Dataset,
        context: StageContext,
    ) -> ray.data.Dataset:
        """
        Execute the stage on a dataset.

        Args:
            dataset: Input Ray Dataset.
            context: Execution context with caching, manifests, etc.

        Returns:
            Transformed Ray Dataset.
        """
        ...

    def validate_input(self, dataset: ray.data.Dataset) -> list[str]:
        """
        Validate that dataset has required input fields.

        Args:
            dataset: Input dataset.

        Returns:
            List of validation error messages.
        """
        errors = []
        schema = dataset.schema()

        if schema is None:
            return ["Dataset has no schema"]

        field_names = {f.name for f in schema}
        for required_field in self.input_schema:
            if required_field not in field_names:
                errors.append(f"Missing required field: {required_field}")

        return errors

    def validate_output(self, dataset: ray.data.Dataset) -> list[str]:
        """
        Validate that output dataset has expected fields.

        Args:
            dataset: Output dataset.

        Returns:
            List of validation error messages.
        """
        errors = []
        schema = dataset.schema()

        if schema is None:
            return ["Dataset has no schema"]

        field_names = {f.name for f in schema}
        for expected_field in self.output_schema:
            if expected_field not in field_names:
                errors.append(f"Missing expected output field: {expected_field}")

        return errors


@dataclass
class StageContext:
    """
    Execution context for a stage.

    Provides access to caching, manifests, and run configuration.
    """

    run_id: str
    stage_index: int
    output_dir: str

    # Fingerprints
    prompt_hash: str | None = None
    model_hash: str | None = None
    sampling_params_hash: str | None = None

    # Seeds
    stage_seed: int | None = None

    # Caching
    cache_enabled: bool = True
    cache_store: Any = None  # Optional CacheStore

    # Manifests
    manifest_writer: Any = None  # Optional ManifestWriter

    # Metrics
    metrics: dict[str, Any] = field(default_factory=dict)

    def get_output_path(self, suffix: str = "") -> str:
        """Get output path for this stage."""
        return f"{self.output_dir}/{suffix}" if suffix else self.output_dir
