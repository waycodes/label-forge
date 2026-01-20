"""
Synthetic data specification.

Defines schemas for synthetic record types and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SynthRecordType(str, Enum):
    """Types of synthetic data records."""

    CAPTION_VARIANT = "caption_variant"
    QA_PAIR = "qa_pair"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    PARAPHRASE = "paraphrase"
    SUMMARY = "summary"


class SynthMetadata(BaseModel):
    """Metadata for a synthetic record."""

    model_config = ConfigDict(frozen=True)

    # Generation info
    record_type: SynthRecordType = Field(description="Type of synthetic record")
    generator_model: str = Field(description="Model used for generation")
    generator_prompt_hash: str | None = Field(
        default=None, description="Hash of prompt used"
    )
    generation_seed: int | None = Field(default=None, description="Seed used")

    # Provenance
    source_row_ids: list[str] = Field(
        default_factory=list, description="Row IDs used as input"
    )
    source_stage: str | None = Field(
        default=None, description="Stage that produced source data"
    )

    # Quality
    quality_score: float | None = Field(
        default=None, description="Optional quality score"
    )
    filtered: bool = Field(default=False, description="Whether filtered out")
    filter_reason: str | None = Field(default=None, description="Reason for filtering")

    # Timestamps
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class SynthRecord(BaseModel):
    """A synthetic data record."""

    model_config = ConfigDict(frozen=True)

    # Identity
    row_id: str = Field(description="Unique ID for this synthetic record")

    # Content fields (depend on record type)
    content: dict[str, Any] = Field(
        default_factory=dict, description="Synthetic content"
    )

    # Metadata
    metadata: SynthMetadata = Field(description="Generation metadata")

    @property
    def record_type(self) -> SynthRecordType:
        """Get record type from metadata."""
        return self.metadata.record_type


class QAPair(BaseModel):
    """A question-answer pair."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(description="Generated question")
    answer: str = Field(description="Expected answer")
    context: str | None = Field(default=None, description="Optional context")


class Instruction(BaseModel):
    """An instruction-response pair."""

    model_config = ConfigDict(frozen=True)

    instruction: str = Field(description="User instruction")
    response: str = Field(description="Assistant response")
    system_prompt: str | None = Field(default=None, description="Optional system prompt")


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    model_config = ConfigDict(frozen=True)

    role: str = Field(description="Speaker role (user, assistant, system)")
    content: str = Field(description="Turn content")
    image_ref: str | None = Field(default=None, description="Optional image reference")


class Conversation(BaseModel):
    """A multi-turn conversation."""

    model_config = ConfigDict(frozen=True)

    turns: list[ConversationTurn] = Field(description="Conversation turns")
    image_row_id: str | None = Field(
        default=None, description="Row ID of source image"
    )

    @property
    def turn_count(self) -> int:
        """Number of turns in conversation."""
        return len(self.turns)


@dataclass
class SynthSpec:
    """Specification for synthetic data generation."""

    record_type: SynthRecordType
    model_name: str
    prompt_template: str
    output_schema: dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 1024
    temperature: float = 0.7
    num_variants: int = 1
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_type": self.record_type.value,
            "model_name": self.model_name,
            "prompt_template": self.prompt_template,
            "output_schema": self.output_schema,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "num_variants": self.num_variants,
            "seed": self.seed,
        }


def create_synth_row_id(
    source_row_id: str,
    record_type: SynthRecordType,
    variant_index: int = 0,
) -> str:
    """
    Create a stable row_id for synthetic data.

    Args:
        source_row_id: Row ID of source data.
        record_type: Type of synthetic record.
        variant_index: Index for multiple variants.

    Returns:
        Unique synthetic row ID.
    """
    import xxhash

    content = f"{source_row_id}:{record_type.value}:{variant_index}"
    hash_val = xxhash.xxh64(content.encode()).hexdigest()[:12]
    return f"synth_{hash_val}"
