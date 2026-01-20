"""
Tests for VLM tags, text classification, and text extraction stages.
"""

from __future__ import annotations

import pytest

from labelforge.pipelines.stage import StageConfig, StageContext, StageType
from labelforge.pipelines.stages.text_extract import (
    TextExtractStage,
    create_text_extract_stage,
)
from labelforge.pipelines.stages.text_label import (
    TextLabelStage,
    create_text_label_stage,
)
from labelforge.pipelines.stages.vlm_tags import (
    VLMTagsStage,
    create_vlm_tags_stage,
)


class TestVLMTagsStage:
    """Tests for VLMTagsStage."""

    def test_create_stage_defaults(self) -> None:
        """Test creating VLM tags stage with defaults."""
        stage = create_vlm_tags_stage()

        assert stage.name == "vlm_tags"
        assert stage.stage_type == StageType.VLM_TAGS
        assert stage.max_tags == 10
        assert stage.use_guided_decoding is True

    def test_create_stage_with_vocabulary(self) -> None:
        """Test creating VLM tags stage with closed vocabulary."""
        vocab = ["landscape", "portrait", "nature", "urban", "abstract"]
        stage = create_vlm_tags_stage(
            name="closed_tagger",
            tag_vocabulary=vocab,
            max_tags=5,
        )

        assert stage.name == "closed_tagger"
        assert stage.tag_vocabulary == vocab
        assert stage.max_tags == 5

    def test_create_stage_with_attributes(self) -> None:
        """Test creating VLM tags stage with attribute schema."""
        attr_schema = {
            "style": ["photo", "illustration", "painting"],
            "mood": ["happy", "sad", "neutral"],
            "setting": [],  # Free-form
        }
        stage = create_vlm_tags_stage(
            attribute_schema=attr_schema,
        )

        assert stage.attribute_schema == attr_schema

    def test_input_schema(self) -> None:
        """Test input schema requirements."""
        stage = create_vlm_tags_stage()

        assert "image" in stage.input_schema
        assert "row_id" in stage.input_schema

    def test_output_schema(self) -> None:
        """Test output schema fields."""
        stage = create_vlm_tags_stage()

        assert "tags" in stage.output_schema
        assert "attributes" in stage.output_schema
        assert "tags_raw" in stage.output_schema

    def test_build_prompt_open_vocabulary(self) -> None:
        """Test prompt building with open vocabulary."""
        stage = create_vlm_tags_stage(max_tags=5)
        prompt = stage._build_tagging_prompt()

        assert "Up to 5" in prompt
        assert "JSON" in prompt

    def test_build_prompt_closed_vocabulary(self) -> None:
        """Test prompt building with closed vocabulary."""
        vocab = ["cat", "dog", "bird"]
        stage = create_vlm_tags_stage(tag_vocabulary=vocab)
        prompt = stage._build_tagging_prompt()

        assert "cat" in prompt
        assert "dog" in prompt
        assert "bird" in prompt

    def test_output_schema_structure(self) -> None:
        """Test JSON output schema structure."""
        vocab = ["a", "b"]
        attr_schema = {"size": ["small", "large"]}
        stage = create_vlm_tags_stage(
            tag_vocabulary=vocab,
            attribute_schema=attr_schema,
        )

        schema = stage._get_output_schema()

        assert schema["type"] == "object"
        assert "tags" in schema["properties"]
        assert "attributes" in schema["properties"]

        # Check tags enum
        tags_items = schema["properties"]["tags"]["items"]
        assert tags_items.get("enum") == vocab

    def test_postprocess_valid_json(self) -> None:
        """Test postprocessing with valid JSON."""
        stage = create_vlm_tags_stage()
        postprocess_fn = stage._create_postprocess_fn()

        row = {
            "row_id": "123",
            "generated_text": '{"tags": ["nature", "sunset"], "attributes": {"mood": "calm"}}',
        }
        result = postprocess_fn(row)

        assert result["tags"] == ["nature", "sunset"]
        assert result["attributes"] == {"mood": "calm"}
        assert "parse_error" not in result

    def test_postprocess_invalid_json(self) -> None:
        """Test postprocessing with invalid JSON."""
        stage = create_vlm_tags_stage()
        postprocess_fn = stage._create_postprocess_fn()

        row = {
            "row_id": "123",
            "generated_text": "not valid json",
        }
        result = postprocess_fn(row)

        assert result["tags"] == []
        assert result["attributes"] == {}
        assert result["parse_error"] is True

    def test_fingerprint_stability(self) -> None:
        """Test stage fingerprint is stable."""
        stage1 = create_vlm_tags_stage(name="test", max_tags=5)
        stage2 = create_vlm_tags_stage(name="test", max_tags=5)
        stage3 = create_vlm_tags_stage(name="test", max_tags=10)

        assert stage1.fingerprint() == stage2.fingerprint()
        assert stage1.fingerprint() != stage3.fingerprint()


class TestTextLabelStage:
    """Tests for TextLabelStage."""

    @pytest.fixture
    def categories(self) -> list[str]:
        """Sample categories for testing."""
        return ["positive", "negative", "neutral"]

    def test_create_stage_defaults(self, categories: list[str]) -> None:
        """Test creating text label stage with defaults."""
        stage = create_text_label_stage(categories=categories)

        assert stage.name == "text_label"
        assert stage.stage_type == StageType.TEXT_LABEL
        assert stage.categories == categories
        assert stage.multi_label is False

    def test_create_stage_multi_label(self, categories: list[str]) -> None:
        """Test creating multi-label stage."""
        stage = create_text_label_stage(
            name="multi_classifier",
            categories=categories,
            multi_label=True,
            max_labels=2,
        )

        assert stage.multi_label is True
        assert stage.max_labels == 2

    def test_create_stage_with_reasoning(self, categories: list[str]) -> None:
        """Test creating stage with reasoning."""
        stage = create_text_label_stage(
            categories=categories,
            include_reasoning=True,
        )

        assert stage.include_reasoning is True
        assert "reasoning" in stage.output_schema

    def test_input_schema(self, categories: list[str]) -> None:
        """Test input schema requirements."""
        stage = create_text_label_stage(categories=categories, text_field="content")

        assert "content" in stage.input_schema
        assert "row_id" in stage.input_schema

    def test_output_schema(self, categories: list[str]) -> None:
        """Test output schema fields."""
        stage = create_text_label_stage(categories=categories)

        assert "labels" in stage.output_schema
        assert "label_primary" in stage.output_schema
        assert "label_raw" in stage.output_schema

    def test_build_prompt(self, categories: list[str]) -> None:
        """Test classification prompt building."""
        stage = create_text_label_stage(categories=categories)
        prompt = stage._build_classification_prompt("This is great!")

        assert "positive" in prompt
        assert "negative" in prompt
        assert "neutral" in prompt
        assert "This is great!" in prompt

    def test_build_prompt_multi_label(self, categories: list[str]) -> None:
        """Test multi-label prompt building."""
        stage = create_text_label_stage(
            categories=categories,
            multi_label=True,
            max_labels=3,
        )
        prompt = stage._build_classification_prompt("Test text")

        assert "one or more categories" in prompt

    def test_output_schema_single_label(self, categories: list[str]) -> None:
        """Test output schema for single-label mode."""
        stage = create_text_label_stage(categories=categories, multi_label=False)
        schema = stage._get_output_schema()

        labels_schema = schema["properties"]["labels"]
        assert labels_schema["maxItems"] == 1

    def test_output_schema_multi_label(self, categories: list[str]) -> None:
        """Test output schema for multi-label mode."""
        stage = create_text_label_stage(
            categories=categories,
            multi_label=True,
            max_labels=3,
        )
        schema = stage._get_output_schema()

        labels_schema = schema["properties"]["labels"]
        assert labels_schema["maxItems"] == 3

    def test_postprocess_valid(self, categories: list[str]) -> None:
        """Test postprocessing with valid JSON."""
        stage = create_text_label_stage(categories=categories)
        postprocess_fn = stage._create_postprocess_fn()

        row = {
            "row_id": "123",
            "generated_text": '{"labels": ["positive"], "primary": "positive"}',
        }
        result = postprocess_fn(row)

        assert result["labels"] == ["positive"]
        assert result["label_primary"] == "positive"
        assert "parse_error" not in result

    def test_postprocess_with_reasoning(self, categories: list[str]) -> None:
        """Test postprocessing with reasoning."""
        stage = create_text_label_stage(
            categories=categories,
            include_reasoning=True,
        )
        postprocess_fn = stage._create_postprocess_fn()

        row = {
            "row_id": "123",
            "generated_text": '{"labels": ["positive"], "primary": "positive", "reasoning": "Good vibes"}',
        }
        result = postprocess_fn(row)

        assert result["reasoning"] == "Good vibes"

    def test_fingerprint_stability(self, categories: list[str]) -> None:
        """Test stage fingerprint is stable."""
        stage1 = create_text_label_stage(name="test", categories=categories)
        stage2 = create_text_label_stage(name="test", categories=categories)
        stage3 = create_text_label_stage(
            name="test", categories=categories, multi_label=True
        )

        assert stage1.fingerprint() == stage2.fingerprint()
        assert stage1.fingerprint() != stage3.fingerprint()


class TestTextExtractStage:
    """Tests for TextExtractStage."""

    @pytest.fixture
    def extraction_schema(self) -> dict:
        """Sample extraction schema."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "email": {"type": "string", "description": "Email address"},
                "phone": {"type": "string", "description": "Phone number"},
            },
            "required": ["name"],
        }

    def test_create_stage_defaults(self, extraction_schema: dict) -> None:
        """Test creating text extract stage with defaults."""
        stage = create_text_extract_stage(extraction_schema=extraction_schema)

        assert stage.name == "text_extract"
        assert stage.stage_type == StageType.TEXT_EXTRACT
        assert stage.extraction_schema == extraction_schema

    def test_create_stage_custom_field(self, extraction_schema: dict) -> None:
        """Test creating stage with custom text field."""
        stage = create_text_extract_stage(
            extraction_schema=extraction_schema,
            text_field="content",
        )

        assert stage.text_field == "content"
        assert "content" in stage.input_schema

    def test_input_schema(self, extraction_schema: dict) -> None:
        """Test input schema requirements."""
        stage = create_text_extract_stage(extraction_schema=extraction_schema)

        assert "text" in stage.input_schema
        assert "row_id" in stage.input_schema

    def test_output_schema(self, extraction_schema: dict) -> None:
        """Test output schema fields."""
        stage = create_text_extract_stage(extraction_schema=extraction_schema)

        assert "extracted" in stage.output_schema
        assert "extraction_raw" in stage.output_schema

    def test_build_prompt(self, extraction_schema: dict) -> None:
        """Test extraction prompt building."""
        stage = create_text_extract_stage(extraction_schema=extraction_schema)
        prompt = stage._build_extraction_prompt("John Doe, john@example.com, 555-1234")

        assert "name" in prompt
        assert "email" in prompt
        assert "phone" in prompt
        assert "John Doe" in prompt

    def test_describe_schema(self, extraction_schema: dict) -> None:
        """Test schema description generation."""
        stage = create_text_extract_stage(extraction_schema=extraction_schema)
        description = stage._describe_schema(extraction_schema)

        assert "name" in description
        assert "required" in description
        assert "optional" in description

    def test_postprocess_valid(self, extraction_schema: dict) -> None:
        """Test postprocessing with valid JSON."""
        stage = create_text_extract_stage(extraction_schema=extraction_schema)
        postprocess_fn = stage._create_postprocess_fn()

        row = {
            "row_id": "123",
            "generated_text": '{"name": "John Doe", "email": "john@example.com"}',
        }
        result = postprocess_fn(row)

        assert result["extracted"]["name"] == "John Doe"
        assert result["extracted"]["email"] == "john@example.com"
        assert "parse_error" not in result

    def test_postprocess_invalid(self, extraction_schema: dict) -> None:
        """Test postprocessing with invalid JSON."""
        stage = create_text_extract_stage(extraction_schema=extraction_schema)
        postprocess_fn = stage._create_postprocess_fn()

        row = {
            "row_id": "123",
            "generated_text": "not json",
        }
        result = postprocess_fn(row)

        assert result["extracted"] == {}
        assert result["parse_error"] is True

    def test_fingerprint_stability(self, extraction_schema: dict) -> None:
        """Test stage fingerprint is stable."""
        stage1 = create_text_extract_stage(
            name="test", extraction_schema=extraction_schema
        )
        stage2 = create_text_extract_stage(
            name="test", extraction_schema=extraction_schema
        )

        different_schema = {
            "type": "object",
            "properties": {"other": {"type": "string"}},
        }
        stage3 = create_text_extract_stage(
            name="test", extraction_schema=different_schema
        )

        assert stage1.fingerprint() == stage2.fingerprint()
        assert stage1.fingerprint() != stage3.fingerprint()

    def test_nested_schema(self) -> None:
        """Test with nested extraction schema."""
        nested_schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string"},
                        "last_name": {"type": "string"},
                    },
                },
                "company": {"type": "string"},
            },
        }
        stage = create_text_extract_stage(extraction_schema=nested_schema)
        description = stage._describe_schema(nested_schema)

        assert "person" in description
        assert "object with" in description


class TestStageTypeConsistency:
    """Tests for stage type consistency."""

    def test_all_stages_have_distinct_types(self) -> None:
        """Test that all stage types are distinct."""
        vlm_tags = create_vlm_tags_stage()
        text_label = create_text_label_stage(categories=["a", "b"])
        text_extract = create_text_extract_stage(
            extraction_schema={"type": "object", "properties": {}}
        )

        types = {
            vlm_tags.stage_type,
            text_label.stage_type,
            text_extract.stage_type,
        }

        assert len(types) == 3  # All unique

    def test_stages_have_correct_types(self) -> None:
        """Test stages have expected types."""
        vlm_tags = create_vlm_tags_stage()
        text_label = create_text_label_stage(categories=["a"])
        text_extract = create_text_extract_stage(
            extraction_schema={"type": "object", "properties": {}}
        )

        assert vlm_tags.stage_type == StageType.VLM_TAGS
        assert text_label.stage_type == StageType.TEXT_LABEL
        assert text_extract.stage_type == StageType.TEXT_EXTRACT
