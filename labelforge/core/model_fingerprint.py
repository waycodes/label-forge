"""
Model fingerprinting for cache keys and manifests.

Computes stable hashes of model identity and configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import xxhash

from labelforge.core.json_canonical import canonical_json_bytes

if TYPE_CHECKING:
    from labelforge.core.model_spec import ModelSpec, SamplingParams


def compute_model_hash(model_spec: ModelSpec) -> str:
    """
    Compute hash of a model specification.

    The hash includes all fields that affect model behavior:
    - Model source and revision
    - Task type
    - Quantization and dtype
    - Parallelism settings
    - Engine kwargs

    Args:
        model_spec: The model specification to hash.

    Returns:
        Hex-encoded hash string.
    """
    content = {
        "model_source": model_spec.model_source,
        "revision": model_spec.revision,
        "task_type": model_spec.task_type.value,
        "quantization": model_spec.quantization.value,
        "dtype": model_spec.dtype.value,
        "max_model_len": model_spec.max_model_len,
        "tensor_parallel_size": model_spec.tensor_parallel_size,
        "pipeline_parallel_size": model_spec.pipeline_parallel_size,
        "has_image": model_spec.has_image,
        "engine_kwargs": model_spec.engine_kwargs,
    }

    json_bytes = canonical_json_bytes(content)
    return xxhash.xxh64(json_bytes).hexdigest()


def compute_sampling_params_hash(params: SamplingParams) -> str:
    """
    Compute hash of sampling parameters.

    Args:
        params: The sampling parameters to hash.

    Returns:
        Hex-encoded hash string.
    """
    content = {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "max_tokens": params.max_tokens,
        "min_tokens": params.min_tokens,
        "frequency_penalty": params.frequency_penalty,
        "presence_penalty": params.presence_penalty,
        "repetition_penalty": params.repetition_penalty,
        "seed": params.seed,
        "guided_decoding": params.guided_decoding,
        "stop": params.stop,
        "stop_token_ids": params.stop_token_ids,
    }

    json_bytes = canonical_json_bytes(content)
    return xxhash.xxh64(json_bytes).hexdigest()


def compute_inference_hash(
    model_spec: ModelSpec,
    sampling_params: SamplingParams,
) -> str:
    """
    Compute combined hash for an inference configuration.

    Args:
        model_spec: The model specification.
        sampling_params: The sampling parameters.

    Returns:
        Hex-encoded hash string.
    """
    model_hash = compute_model_hash(model_spec)
    params_hash = compute_sampling_params_hash(sampling_params)

    combined = f"{model_hash}:{params_hash}"
    return xxhash.xxh64(combined.encode("utf-8")).hexdigest()


class ModelFingerprint:
    """
    Complete fingerprint for a model configuration.

    Used for cache keys and manifest records.
    """

    def __init__(
        self,
        model_source: str,
        revision: str | None,
        task_type: str,
        model_hash: str,
        sampling_params_hash: str | None = None,
    ):
        self.model_source = model_source
        self.revision = revision
        self.task_type = task_type
        self.model_hash = model_hash
        self.sampling_params_hash = sampling_params_hash

    @property
    def combined_hash(self) -> str:
        """Compute combined hash of model and sampling params."""
        components = [
            self.model_hash,
            self.sampling_params_hash or "",
        ]
        combined = "|".join(components)
        return xxhash.xxh64(combined.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_source": self.model_source,
            "revision": self.revision,
            "task_type": self.task_type,
            "model_hash": self.model_hash,
            "sampling_params_hash": self.sampling_params_hash,
            "combined_hash": self.combined_hash,
        }

    @classmethod
    def from_spec(
        cls,
        model_spec: ModelSpec,
        sampling_params: SamplingParams | None = None,
    ) -> ModelFingerprint:
        """
        Create fingerprint from model spec and optional sampling params.

        Args:
            model_spec: The model specification.
            sampling_params: Optional sampling parameters.

        Returns:
            ModelFingerprint instance.
        """
        model_hash = compute_model_hash(model_spec)
        params_hash = (
            compute_sampling_params_hash(sampling_params) if sampling_params else None
        )

        return cls(
            model_source=model_spec.model_source,
            revision=model_spec.revision,
            task_type=model_spec.task_type.value,
            model_hash=model_hash,
            sampling_params_hash=params_hash,
        )
