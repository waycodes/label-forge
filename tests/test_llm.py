"""Tests for LLM integration modules."""

import pytest

from labelforge.llm.determinism import (
    DeterminismMode,
    DeterminismConfig,
    get_reproducibility_warnings,
    create_runtime_env_for_determinism,
)
from labelforge.llm.vlm_messages import (
    build_vlm_messages,
    build_vlm_messages_with_examples,
    build_vlm_content,
    extract_text_from_messages,
)
from labelforge.llm.sampling import (
    inject_sampling_params,
    create_sampling_injector,
    merge_sampling_params,
    get_sampling_params_hash,
)
from labelforge.llm.postprocess import (
    parse_json_output,
    clean_output_text,
    validate_output_schema,
    create_postprocess_with_parsing,
)
from labelforge.core.model_spec import SamplingParams


class TestDeterminismMode:
    """Tests for determinism modes."""

    def test_batch_invariant_config(self):
        """Batch invariant mode should set correct env vars."""
        config = DeterminismConfig(mode=DeterminismMode.BATCH_INVARIANT)
        assert config.env_vars == {"VLLM_BATCH_INVARIANT": "1"}

    def test_deterministic_scheduling_config(self):
        """Deterministic scheduling should disable multiprocessing."""
        config = DeterminismConfig(mode=DeterminismMode.DETERMINISTIC_SCHEDULING)
        assert config.env_vars == {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}

    def test_none_mode(self):
        """None mode should have no env vars."""
        config = DeterminismConfig(mode=DeterminismMode.NONE)
        assert config.env_vars == {}

    def test_reproducibility_warnings(self):
        """Should return warnings for each mode."""
        warnings = get_reproducibility_warnings(DeterminismMode.NONE)
        assert any("no determinism" in w.lower() for w in warnings)

    def test_runtime_env_creation(self):
        """Should create runtime env with determinism settings."""
        runtime_env = create_runtime_env_for_determinism(
            DeterminismMode.BATCH_INVARIANT,
            base_runtime_env={"pip": ["numpy"]},
        )
        assert runtime_env["env_vars"]["VLLM_BATCH_INVARIANT"] == "1"
        assert runtime_env["pip"] == ["numpy"]


class TestVLMMessages:
    """Tests for VLM message builder."""

    def test_build_text_only(self):
        """Text-only messages should work."""
        messages = build_vlm_messages("Describe this.")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Describe this."

    def test_build_with_system(self):
        """System instruction should be added."""
        messages = build_vlm_messages(
            "Describe this.",
            system_instruction="You are a captioner."
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_build_with_examples(self):
        """Few-shot examples should be added."""
        examples = [
            {"input": "Example input", "output": "Example output"},
        ]
        messages = build_vlm_messages_with_examples(
            "Real input",
            examples=examples,
        )
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Example input"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_build_vlm_content(self):
        """Content list should work."""
        content = build_vlm_content("Description text")
        assert len(content) == 1
        assert content[0]["type"] == "text"

    def test_extract_text_from_messages(self):
        """Text extraction should work."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        text = extract_text_from_messages(messages)
        assert "[system]" in text.lower()
        assert "Hello" in text
        assert "Hi there!" in text


class TestSamplingInjection:
    """Tests for sampling parameter injection."""

    def test_inject_sampling_params(self):
        """Should inject params into row."""
        params = SamplingParams(temperature=0.5, max_tokens=100)
        row = {"text": "Hello", "row_id": "123"}
        
        result = inject_sampling_params(row, params)
        assert "sampling_params" in result
        assert result["sampling_params"]["temperature"] == 0.5

    def test_inject_with_row_seed(self):
        """Should inject row-specific seed."""
        params = SamplingParams(temperature=0.5)
        row = {"text": "Hello", "row_id": "123"}
        
        result = inject_sampling_params(row, params, row_seed=42)
        assert result["sampling_params"]["seed"] == 42

    def test_inject_with_overrides(self):
        """Should apply overrides."""
        params = SamplingParams(temperature=0.5)
        row = {"text": "Hello"}
        
        result = inject_sampling_params(row, params, overrides={"temperature": 0.0})
        assert result["sampling_params"]["temperature"] == 0.0

    def test_create_sampling_injector(self):
        """Should create injectable function."""
        params = SamplingParams(temperature=0.7)
        inject = create_sampling_injector(params)
        
        row = {"text": "Hello", "row_id": "123"}
        result = inject(row)
        assert "sampling_params" in result

    def test_merge_sampling_params(self):
        """Should merge params correctly."""
        base = {"temperature": 0.7, "max_tokens": 100}
        overrides = {"temperature": 0.0}
        
        result = merge_sampling_params(base, overrides)
        assert result["temperature"] == 0.0
        assert result["max_tokens"] == 100

    def test_get_sampling_params_hash(self):
        """Should hash params deterministically."""
        params = {"temperature": 0.5, "max_tokens": 100}
        hash1 = get_sampling_params_hash(params)
        hash2 = get_sampling_params_hash(params)
        assert hash1 == hash2


class TestPostprocessing:
    """Tests for postprocessing and parsing."""

    def test_parse_json_output_simple(self):
        """Simple JSON should parse."""
        result = parse_json_output('{"key": "value"}')
        assert result.success is True
        assert result.data == {"key": "value"}

    def test_parse_json_output_markdown(self):
        """JSON in markdown code block should parse."""
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_output(text)
        assert result.success is True
        assert result.data == {"key": "value"}

    def test_parse_json_output_invalid(self):
        """Invalid JSON should return error."""
        result = parse_json_output('not valid json')
        assert result.success is False
        assert "parse error" in result.error.lower()

    def test_clean_output_text(self):
        """Should clean common artifacts."""
        text = "  Answer: The result is 42  "
        result = clean_output_text(text)
        assert result == "The result is 42"

    def test_validate_output_schema(self):
        """Should validate required fields."""
        data = {"field1": "value", "field2": 42}
        valid, missing = validate_output_schema(data, ["field1", "field2"])
        assert valid is True
        assert missing == []

    def test_validate_output_schema_missing(self):
        """Should detect missing fields."""
        data = {"field1": "value"}
        valid, missing = validate_output_schema(data, ["field1", "field2"])
        assert valid is False
        assert "field2" in missing

    def test_create_postprocess_with_parsing(self):
        """Should create parsing postprocessor."""
        postprocess = create_postprocess_with_parsing(output_field="result")
        
        row = {"generated_text": '{"score": 5}', "row_id": "123"}
        result = postprocess(row)
        assert result["result"] == {"score": 5}
        assert result["row_id"] == "123"
