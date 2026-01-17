"""
Canonical row schema for multimodal items.

Every pipeline stage uses these schemas for consistent data representation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DataSourceType(str, Enum):
    """Type of data source for provenance tracking."""

    FILE = "file"
    DATASET = "dataset"
    URL = "url"
    SYNTHETIC = "synthetic"


class DataSource(BaseModel):
    """Provenance information for a data row."""

    model_config = ConfigDict(frozen=True)

    source_type: DataSourceType = Field(description="Type of the data source")
    source_uri: str = Field(description="URI or identifier of the source")
    source_key: str | None = Field(
        default=None, description="Key within the source (e.g., row index, file path)"
    )
    source_version: str | None = Field(
        default=None, description="Version or revision of the source"
    )


class RowMetadata(BaseModel):
    """
    Extensible metadata container for a data row.

    Use this for any additional fields not part of the core schema.
    """

    model_config = ConfigDict(extra="allow")

    labels: dict[str, Any] = Field(
        default_factory=dict, description="Ground truth or predicted labels"
    )
    annotations: dict[str, Any] = Field(
        default_factory=dict, description="Human or model annotations"
    )
    scores: dict[str, float] = Field(
        default_factory=dict, description="Quality or relevance scores"
    )
    custom: dict[str, Any] = Field(
        default_factory=dict, description="Custom fields for domain-specific data"
    )


class MultimodalRow(BaseModel):
    """
    Canonical schema for a multimodal data row.

    This is the primary data unit flowing through LabelForge pipelines.
    Images are stored by reference (URI) and loaded on demand.
    """

    model_config = ConfigDict(frozen=True)

    # Core identity
    row_id: str = Field(
        description="Stable, content-derived identifier for this row. "
        "Used as cache key and manifest reference."
    )

    # Image data (optional for text-only rows)
    image_uri: str | None = Field(
        default=None,
        description="URI to the image (file://, s3://, gs://, http://). "
        "Images are loaded on demand during processing.",
    )
    image_hash: str | None = Field(
        default=None,
        description="Content hash of the image for cache invalidation. "
        "Computed from image bytes using xxhash.",
    )

    # Text data (optional for image-only rows)
    text: str | None = Field(
        default=None, description="Primary text content (caption, instruction, etc.)"
    )
    text_secondary: str | None = Field(
        default=None, description="Secondary text (response, answer, etc.)"
    )

    # Provenance
    source: DataSource = Field(description="Origin information for this row")

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Row creation timestamp"
    )
    updated_at: datetime | None = Field(
        default=None, description="Last update timestamp"
    )

    # Extensible metadata
    metadata: RowMetadata = Field(
        default_factory=RowMetadata, description="Additional metadata and annotations"
    )


class StageOutput(BaseModel):
    """
    Output record from a pipeline stage.

    Wraps the input row with stage-specific outputs and audit information.
    """

    model_config = ConfigDict(frozen=True)

    # Link to input
    row_id: str = Field(description="Row ID from input")

    # Stage identification
    stage_name: str = Field(description="Name of the stage that produced this output")
    stage_version: str = Field(description="Version of the stage")

    # Output data
    output: dict[str, Any] = Field(description="Stage-specific output data")

    # Audit trail
    prompt_hash: str = Field(description="Hash of the prompt used")
    model_hash: str = Field(description="Hash of the model config used")
    sampling_params_hash: str = Field(description="Hash of sampling parameters")

    # Timing
    latency_ms: float = Field(description="Processing latency in milliseconds")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Output creation timestamp"
    )

    # Error handling
    error: str | None = Field(default=None, description="Error message if failed")
    raw_response: str | None = Field(
        default=None, description="Raw model response for debugging"
    )


class CaptionOutput(BaseModel):
    """Structured output for image captioning stages."""

    model_config = ConfigDict(frozen=True)

    caption: str = Field(description="Generated caption text")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Model confidence score"
    )


class TagOutput(BaseModel):
    """Structured output for attribute tagging stages."""

    model_config = ConfigDict(frozen=True)

    tags: list[str] = Field(description="List of predicted tags")
    confidences: dict[str, float] | None = Field(
        default=None, description="Per-tag confidence scores"
    )
    rationale: str | None = Field(
        default=None, description="Model's reasoning for tag selection"
    )


class RubricScore(BaseModel):
    """Structured output for rubric scoring stages."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(description="Overall score on the rubric scale")
    normalized_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Score normalized to [0, 1]"
    )
    subscores: dict[str, float] | None = Field(
        default=None, description="Scores for individual rubric criteria"
    )
    explanation: str = Field(description="Justification for the score")


class EmbeddingOutput(BaseModel):
    """Structured output for embedding stages."""

    model_config = ConfigDict(frozen=True)

    embedding: list[float] = Field(description="Embedding vector")
    embedding_model_hash: str = Field(description="Hash of the embedding model config")
    embedding_dim: int = Field(description="Dimensionality of the embedding")
