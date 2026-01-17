"""
Model specification schema.

Fully specifies model identity and runtime parameters for reproducibility.
Mirrors Ray Data LLM vLLMEngineProcessorConfig capabilities.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskType(str, Enum):
    """Type of inference task."""

    GENERATE = "generate"
    EMBED = "embed"
    SCORE = "score"
    REWARD = "reward"
    CLASSIFY = "classify"


class QuantizationType(str, Enum):
    """Quantization method."""

    NONE = "none"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF = "gguf"
    BITSANDBYTES = "bitsandbytes"
    FP8 = "fp8"


class DataType(str, Enum):
    """Model data type."""

    AUTO = "auto"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class ModelSpec(BaseModel):
    """
    Complete specification for a model and its inference configuration.

    This mirrors the capabilities of Ray Data LLM's vLLMEngineProcessorConfig
    while adding fields for reproducibility tracking.
    """

    model_config = ConfigDict(frozen=True)

    # Model identity
    model_source: str = Field(
        description="Model source: HuggingFace ID or URI (s3://, gs://)"
    )
    revision: str | None = Field(
        default=None, description="Model revision/commit hash for pinning"
    )
    task_type: TaskType = Field(
        default=TaskType.GENERATE, description="Type of inference task"
    )

    # Quantization
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE, description="Quantization method"
    )
    dtype: DataType = Field(default=DataType.AUTO, description="Model data type")

    # Model dimensions
    max_model_len: int | None = Field(
        default=None, description="Maximum sequence length"
    )
    max_num_seqs: int | None = Field(
        default=None, description="Maximum number of sequences in a batch"
    )

    # Parallelism
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallelism")
    pipeline_parallel_size: int = Field(
        default=1, ge=1, description="Pipeline parallelism"
    )

    # Ray Data LLM batch settings
    batch_size: int = Field(
        default=64, ge=1, description="Batch size for Ray Data LLM processor"
    )
    concurrency: int = Field(
        default=1, ge=1, description="Number of concurrent model replicas"
    )

    # Multimodal settings
    has_image: bool = Field(default=False, description="Whether model supports images")
    limit_mm_per_prompt: dict[str, int] | None = Field(
        default=None,
        description="Limits on multimodal inputs per prompt (e.g., {'image': 4})",
    )

    # vLLM engine kwargs (passed through to vLLM)
    engine_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional vLLM engine kwargs (e.g., enable_prefix_caching)",
    )

    # Trust settings
    trust_remote_code: bool = Field(
        default=False, description="Whether to trust remote code in model"
    )

    # Caching
    download_dir: str | None = Field(
        default=None, description="Directory for model downloads"
    )

    def to_vllm_config(self) -> dict[str, Any]:
        """
        Convert to vLLMEngineProcessorConfig-compatible dict.

        Returns:
            Dict for use with Ray Data LLM build_processor.
        """
        config: dict[str, Any] = {
            "model": self.model_source,
            "batch_size": self.batch_size,
        }

        if self.revision:
            config["revision"] = self.revision

        if self.task_type != TaskType.GENERATE:
            config["task"] = self.task_type.value

        if self.quantization != QuantizationType.NONE:
            config["quantization"] = self.quantization.value

        if self.dtype != DataType.AUTO:
            config["dtype"] = self.dtype.value

        if self.max_model_len:
            config["max_model_len"] = self.max_model_len

        if self.max_num_seqs:
            config["max_num_seqs"] = self.max_num_seqs

        if self.tensor_parallel_size > 1:
            config["tensor_parallel_size"] = self.tensor_parallel_size

        if self.pipeline_parallel_size > 1:
            config["pipeline_parallel_size"] = self.pipeline_parallel_size

        if self.trust_remote_code:
            config["trust_remote_code"] = True

        if self.download_dir:
            config["download_dir"] = self.download_dir

        # Merge engine kwargs
        engine_kwargs = dict(self.engine_kwargs)

        if self.has_image:
            engine_kwargs["has_image"] = True

        if self.limit_mm_per_prompt:
            engine_kwargs["limit_mm_per_prompt"] = self.limit_mm_per_prompt

        if engine_kwargs:
            config["engine_kwargs"] = engine_kwargs

        return config


class SamplingParams(BaseModel):
    """
    Sampling parameters for text generation.

    These are passed to vLLM for each inference request.
    """

    model_config = ConfigDict(frozen=True)

    # Basic sampling
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling p")
    top_k: int = Field(default=-1, description="Top-k sampling (-1 for disabled)")

    # Output limits
    max_tokens: int = Field(default=256, ge=1, description="Maximum tokens to generate")
    min_tokens: int = Field(default=0, ge=0, description="Minimum tokens to generate")

    # Penalties
    frequency_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    presence_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Presence penalty"
    )
    repetition_penalty: float = Field(
        default=1.0, ge=0.0, description="Repetition penalty"
    )

    # Reproducibility
    seed: int | None = Field(default=None, description="Random seed for sampling")

    # Structured output
    guided_decoding: dict[str, Any] | None = Field(
        default=None,
        description="Guided decoding config (e.g., {'json': json_schema})",
    )

    # Stop conditions
    stop: list[str] | None = Field(default=None, description="Stop sequences")
    stop_token_ids: list[int] | None = Field(
        default=None, description="Stop token IDs"
    )

    def to_vllm_params(self) -> dict[str, Any]:
        """
        Convert to vLLM SamplingParams-compatible dict.

        Returns:
            Dict for use with Ray Data LLM preprocessing.
        """
        params: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        if self.top_k > 0:
            params["top_k"] = self.top_k

        if self.min_tokens > 0:
            params["min_tokens"] = self.min_tokens

        if self.frequency_penalty != 0.0:
            params["frequency_penalty"] = self.frequency_penalty

        if self.presence_penalty != 0.0:
            params["presence_penalty"] = self.presence_penalty

        if self.repetition_penalty != 1.0:
            params["repetition_penalty"] = self.repetition_penalty

        if self.seed is not None:
            params["seed"] = self.seed

        if self.guided_decoding:
            params["guided_decoding"] = self.guided_decoding

        if self.stop:
            params["stop"] = self.stop

        if self.stop_token_ids:
            params["stop_token_ids"] = self.stop_token_ids

        return params


class EmbeddingParams(BaseModel):
    """
    Parameters for embedding generation.
    """

    model_config = ConfigDict(frozen=True)

    pooling_type: str = Field(
        default="mean", description="Pooling type: mean, cls, last"
    )
    normalize: bool = Field(default=True, description="Whether to L2 normalize")


class ScoringParams(BaseModel):
    """
    Parameters for cross-encoder scoring.
    """

    model_config = ConfigDict(frozen=True)

    normalize_scores: bool = Field(
        default=False, description="Whether to normalize scores"
    )
