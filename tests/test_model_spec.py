"""Tests for model specification and fingerprinting."""

import pytest

from labelforge.core.model_spec import (
    ModelSpec,
    SamplingParams,
    EmbeddingParams,
    ScoringParams,
    TaskType,
    QuantizationType,
    DataType,
)
from labelforge.core.model_fingerprint import (
    compute_model_hash,
    compute_sampling_params_hash,
    compute_inference_hash,
    ModelFingerprint,
)


class TestSamplingParams:
    """Tests for SamplingParams model."""

    def test_create_default_params(self):
        """Default params should be creatable."""
        params = SamplingParams()
        assert params.temperature == 0.0  # Default is 0.0 for reproducibility
        assert params.max_tokens == 256

    def test_create_custom_params(self):
        """Custom params should work."""
        params = SamplingParams(
            temperature=0.5,
            max_tokens=1024,
            top_p=0.95,
            top_k=50,
        )
        assert params.temperature == 0.5
        assert params.max_tokens == 1024

    def test_greedy_decoding(self):
        """Temperature=0 for greedy decoding."""
        params = SamplingParams(temperature=0.0)
        assert params.temperature == 0.0

    def test_with_seed(self):
        """Seed should be settable."""
        params = SamplingParams(seed=42)
        assert params.seed == 42

    def test_to_vllm_params(self):
        """Should convert to vLLM dict format."""
        params = SamplingParams(
            temperature=0.5,
            max_tokens=100,
            seed=42,
        )
        vllm_params = params.to_vllm_params()
        assert vllm_params["temperature"] == 0.5
        assert vllm_params["max_tokens"] == 100
        assert vllm_params["seed"] == 42


class TestEmbeddingParams:
    """Tests for EmbeddingParams model."""

    def test_create_embedding_params(self):
        """Embedding params should be creatable."""
        params = EmbeddingParams(
            pooling_type="mean",
            normalize=True,
        )
        assert params.pooling_type == "mean"
        assert params.normalize is True

    def test_default_values(self):
        """Default values should be set."""
        params = EmbeddingParams()
        assert params.normalize is True


class TestScoringParams:
    """Tests for ScoringParams model."""

    def test_create_scoring_params(self):
        """Scoring params should be creatable."""
        params = ScoringParams(
            normalize_scores=True,
        )
        assert params.normalize_scores is True

    def test_default_normalize(self):
        """Default normalize should be False."""
        params = ScoringParams()
        assert params.normalize_scores is False


class TestModelSpec:
    """Tests for ModelSpec model."""

    def test_create_basic_spec(self):
        """Basic spec should be creatable."""
        spec = ModelSpec(
            model_source="meta-llama/Llama-2-7b-hf",
        )
        assert spec.model_source == "meta-llama/Llama-2-7b-hf"
        assert spec.task_type == TaskType.GENERATE

    def test_create_vlm_spec(self):
        """VLM spec should be creatable."""
        spec = ModelSpec(
            model_source="llava-hf/llava-1.5-7b-hf",
            task_type=TaskType.GENERATE,
            has_image=True,
        )
        assert spec.has_image is True

    def test_create_embedding_spec(self):
        """Embedding spec should be creatable."""
        spec = ModelSpec(
            model_source="BAAI/bge-base-en-v1.5",
            task_type=TaskType.EMBED,
        )
        assert spec.task_type == TaskType.EMBED

    def test_create_scoring_spec(self):
        """Scoring spec should be creatable."""
        spec = ModelSpec(
            model_source="cross-encoder/ms-marco-MiniLM-L-6-v2",
            task_type=TaskType.SCORE,
        )
        assert spec.task_type == TaskType.SCORE

    def test_quantization(self):
        """Quantized spec should work."""
        spec = ModelSpec(
            model_source="meta-llama/Llama-2-7b-hf",
            quantization=QuantizationType.GPTQ,
        )
        assert spec.quantization == QuantizationType.GPTQ

    def test_tensor_parallelism(self):
        """Tensor parallelism should be settable."""
        spec = ModelSpec(
            model_source="meta-llama/Llama-2-70b-hf",
            tensor_parallel_size=4,
        )
        assert spec.tensor_parallel_size == 4

    def test_with_revision(self):
        """Model revision should be settable."""
        spec = ModelSpec(
            model_source="meta-llama/Llama-2-7b-hf",
            revision="abc123def",
        )
        assert spec.revision == "abc123def"

    def test_to_vllm_config(self):
        """Should convert to vLLM config dict."""
        spec = ModelSpec(
            model_source="meta-llama/Llama-2-7b-hf",
            tensor_parallel_size=2,
            max_model_len=4096,
        )
        config = spec.to_vllm_config()
        assert config["model"] == "meta-llama/Llama-2-7b-hf"
        assert config["tensor_parallel_size"] == 2
        assert config["max_model_len"] == 4096


class TestModelFingerprint:
    """Tests for model fingerprinting."""

    def test_model_hash_stability(self):
        """Same model should produce same hash."""
        spec = ModelSpec(
            model_source="test/model",
            revision="abc123",
        )
        hash1 = compute_model_hash(spec)
        hash2 = compute_model_hash(spec)
        assert hash1 == hash2

    def test_model_hash_different_revisions(self):
        """Different revisions should produce different hashes."""
        spec1 = ModelSpec(model_source="test/model", revision="rev1")
        spec2 = ModelSpec(model_source="test/model", revision="rev2")
        assert compute_model_hash(spec1) != compute_model_hash(spec2)

    def test_model_hash_different_quantization(self):
        """Different quantization should produce different hashes."""
        spec1 = ModelSpec(model_source="test/model")
        spec2 = ModelSpec(
            model_source="test/model",
            quantization=QuantizationType.AWQ,
        )
        assert compute_model_hash(spec1) != compute_model_hash(spec2)

    def test_sampling_params_hash_stability(self):
        """Same params should produce same hash."""
        params = SamplingParams(temperature=0.5, max_tokens=100)
        hash1 = compute_sampling_params_hash(params)
        hash2 = compute_sampling_params_hash(params)
        assert hash1 == hash2

    def test_sampling_params_hash_different(self):
        """Different params should produce different hashes."""
        p1 = SamplingParams(temperature=0.5)
        p2 = SamplingParams(temperature=0.7)
        assert compute_sampling_params_hash(p1) != compute_sampling_params_hash(p2)

    def test_inference_hash(self):
        """Inference hash should combine model and params."""
        spec = ModelSpec(model_source="test/model")
        params = SamplingParams()
        
        hash1 = compute_inference_hash(spec, params)
        hash2 = compute_inference_hash(spec, params)
        assert hash1 == hash2

    def test_fingerprint_class(self):
        """ModelFingerprint class should work."""
        spec = ModelSpec(model_source="test/model")
        params = SamplingParams()
        
        fp = ModelFingerprint.from_spec(spec, params)
        assert fp.model_hash is not None
        assert fp.combined_hash is not None
