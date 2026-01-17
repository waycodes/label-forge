"""
Row manifest record schema.

Row-level audit trail for full replay and debugging.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RowStatus(str, Enum):
    """Status of a row after processing."""

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"
    CACHED = "cached"


class RowRecord(BaseModel):
    """
    Row-level manifest record.

    Captures all information needed to replay or audit a single row.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    row_id: str = Field(description="Row identifier")
    stage_name: str = Field(description="Stage that processed this row")

    # Input references
    input_hash: str = Field(description="Hash of input data for this row")

    # Processing fingerprints
    prompt_hash: str = Field(description="Hash of prompt used")
    model_hash: str = Field(description="Hash of model config used")
    sampling_params_hash: str = Field(description="Hash of sampling parameters")
    seed: int | None = Field(default=None, description="Seed used for this row")

    # Output references
    output_hash: str | None = Field(
        default=None, description="Hash of output data"
    )
    output_ref: str | None = Field(
        default=None, description="Reference to output (path or cache key)"
    )

    # Status
    status: RowStatus = Field(description="Processing status")

    # Timing
    latency_ms: float = Field(default=0.0, description="Processing latency in ms")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Record creation timestamp"
    )

    # Error handling
    error_type: str | None = Field(default=None, description="Error type if failed")
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )

    # Raw response (optional, for debugging)
    raw_response: str | None = Field(
        default=None, description="Raw model response"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RowRecord:
        """Create from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def success(
        cls,
        row_id: str,
        stage_name: str,
        input_hash: str,
        prompt_hash: str,
        model_hash: str,
        sampling_params_hash: str,
        output_hash: str,
        output_ref: str,
        latency_ms: float,
        seed: int | None = None,
        raw_response: str | None = None,
    ) -> RowRecord:
        """
        Create a success record.

        Args:
            row_id: Row identifier.
            stage_name: Stage name.
            input_hash: Hash of input data.
            prompt_hash: Hash of prompt.
            model_hash: Hash of model config.
            sampling_params_hash: Hash of sampling params.
            output_hash: Hash of output data.
            output_ref: Reference to output.
            latency_ms: Processing latency.
            seed: Optional seed used.
            raw_response: Optional raw response.

        Returns:
            RowRecord with success status.
        """
        return cls(
            row_id=row_id,
            stage_name=stage_name,
            input_hash=input_hash,
            prompt_hash=prompt_hash,
            model_hash=model_hash,
            sampling_params_hash=sampling_params_hash,
            output_hash=output_hash,
            output_ref=output_ref,
            status=RowStatus.SUCCESS,
            latency_ms=latency_ms,
            seed=seed,
            raw_response=raw_response,
        )

    @classmethod
    def error(
        cls,
        row_id: str,
        stage_name: str,
        input_hash: str,
        prompt_hash: str,
        model_hash: str,
        sampling_params_hash: str,
        error_type: str,
        error_message: str,
        latency_ms: float = 0.0,
        seed: int | None = None,
        raw_response: str | None = None,
    ) -> RowRecord:
        """
        Create an error record.

        Args:
            row_id: Row identifier.
            stage_name: Stage name.
            input_hash: Hash of input data.
            prompt_hash: Hash of prompt.
            model_hash: Hash of model config.
            sampling_params_hash: Hash of sampling params.
            error_type: Type of error.
            error_message: Error message.
            latency_ms: Processing latency before error.
            seed: Optional seed used.
            raw_response: Optional raw response.

        Returns:
            RowRecord with error status.
        """
        return cls(
            row_id=row_id,
            stage_name=stage_name,
            input_hash=input_hash,
            prompt_hash=prompt_hash,
            model_hash=model_hash,
            sampling_params_hash=sampling_params_hash,
            status=RowStatus.ERROR,
            error_type=error_type,
            error_message=error_message,
            latency_ms=latency_ms,
            seed=seed,
            raw_response=raw_response,
        )

    @classmethod
    def cached(
        cls,
        row_id: str,
        stage_name: str,
        input_hash: str,
        prompt_hash: str,
        model_hash: str,
        sampling_params_hash: str,
        output_hash: str,
        output_ref: str,
    ) -> RowRecord:
        """
        Create a cached record.

        Args:
            row_id: Row identifier.
            stage_name: Stage name.
            input_hash: Hash of input data.
            prompt_hash: Hash of prompt.
            model_hash: Hash of model config.
            sampling_params_hash: Hash of sampling params.
            output_hash: Hash of cached output.
            output_ref: Reference to cached output.

        Returns:
            RowRecord with cached status.
        """
        return cls(
            row_id=row_id,
            stage_name=stage_name,
            input_hash=input_hash,
            prompt_hash=prompt_hash,
            model_hash=model_hash,
            sampling_params_hash=sampling_params_hash,
            output_hash=output_hash,
            output_ref=output_ref,
            status=RowStatus.CACHED,
            latency_ms=0.0,
        )

    @classmethod
    def skipped(
        cls,
        row_id: str,
        stage_name: str,
        input_hash: str,
        prompt_hash: str,
        model_hash: str,
        sampling_params_hash: str,
        reason: str,
    ) -> RowRecord:
        """
        Create a skipped record.

        Args:
            row_id: Row identifier.
            stage_name: Stage name.
            input_hash: Hash of input data.
            prompt_hash: Hash of prompt.
            model_hash: Hash of model config.
            sampling_params_hash: Hash of sampling params.
            reason: Reason for skipping.

        Returns:
            RowRecord with skipped status.
        """
        return cls(
            row_id=row_id,
            stage_name=stage_name,
            input_hash=input_hash,
            prompt_hash=prompt_hash,
            model_hash=model_hash,
            sampling_params_hash=sampling_params_hash,
            status=RowStatus.SKIPPED,
            latency_ms=0.0,
            error_message=reason,
        )
