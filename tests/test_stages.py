"""
Integration tests for pipeline stages.

Tests VLM caption, rubric scoring, and stage configuration.
"""

from __future__ import annotations

import pytest

from labelforge.core.prompt_pack import Rubric, RubricCriterion
from labelforge.pipelines.stage import StageConfig, StageContext, StageType
from labelforge.pipelines.stages.rubric_score import (
    RubricScoreStage,
    create_rubric_score_stage,
)
from labelforge.pipelines.stages.vlm_caption import (
    VLMCaptionStage,
    create_vlm_caption_stage,
)


class TestVLMCaptionStage:
    """Tests for VLMCaptionStage."""

    def test_create_vlm_caption_stage_defaults(self) -> None:
        """Test creating VLM caption stage with defaults."""
        stage = create_vlm_caption_stage()

        assert stage.name == "vlm_caption"
        assert stage.stage_type == StageType.VLM_CAPTION
        assert stage.version == "1.0.0"
        assert stage.max_tokens == 256
        assert stage.temperature == 0.0

    def test_create_vlm_caption_stage_custom(self) -> None:
        """Test creating VLM caption stage with custom config."""
        stage = create_vlm_caption_stage(
            name="image_describer",
            system_instruction="You are an image expert.",
            prompt_template="Describe this image briefly.",
            max_tokens=512,
            temperature=0.1,
            version="2.0.0",
        )

        assert stage.name == "image_describer"
        assert stage.version == "2.0.0"
        assert stage.max_tokens == 512
        assert stage.temperature == 0.1
        assert stage.system_instruction == "You are an image expert."

    def test_vlm_caption_stage_input_schema(self) -> None:
        """Test VLM caption stage input schema."""
        stage = create_vlm_caption_stage()

        assert "image" in stage.input_schema
        assert "row_id" in stage.input_schema

    def test_vlm_caption_stage_output_schema(self) -> None:
        """Test VLM caption stage output schema."""
        stage = create_vlm_caption_stage()

        assert "caption" in stage.output_schema
        assert "caption_raw" in stage.output_schema

    def test_vlm_caption_stage_fingerprint(self) -> None:
        """Test that stage fingerprint is stable."""
        stage1 = create_vlm_caption_stage(name="test", max_tokens=256)
        stage2 = create_vlm_caption_stage(name="test", max_tokens=256)
        stage3 = create_vlm_caption_stage(name="test", max_tokens=512)

        assert stage1.fingerprint() == stage2.fingerprint()
        assert stage1.fingerprint() != stage3.fingerprint()

    def test_vlm_caption_preprocess_fn(self) -> None:
        """Test preprocessing function creation."""
        stage = create_vlm_caption_stage(
            system_instruction="Be detailed.",
            prompt_template="Describe: {text}",
        )

        context = StageContext(
            run_id="test_run",
            stage_index=0,
            output_dir="/tmp/output",
        )

        preprocess_fn = stage._create_preprocess_fn(context)

        # Test with sample row
        row = {"row_id": "row_123", "image": None}
        result = preprocess_fn(row)

        assert "messages" in result
        assert "sampling_params" in result
        assert result["row_id"] == "row_123"
        assert result["sampling_params"]["max_tokens"] == 256
        assert result["sampling_params"]["temperature"] == 0.0

    def test_vlm_caption_postprocess_fn(self) -> None:
        """Test postprocessing function creation."""
        stage = create_vlm_caption_stage()
        postprocess_fn = stage._create_postprocess_fn()

        # Test with sample output
        row = {
            "row_id": "row_123",
            "generated_text": "  A beautiful sunset over the ocean.  ",
        }
        result = postprocess_fn(row)

        assert result["row_id"] == "row_123"
        assert result["caption"] == "A beautiful sunset over the ocean."
        assert result["caption_raw"] == "  A beautiful sunset over the ocean.  "


class TestRubricScoreStage:
    """Tests for RubricScoreStage."""

    @pytest.fixture
    def sample_rubric(self) -> Rubric:
        """Create a sample rubric for testing."""
        return Rubric(
            name="text_quality",
            version="1.0",
            description="Evaluates text quality",
            criteria=[
                RubricCriterion(
                    name="clarity",
                    description="How clear is the text",
                    weight=1.0,
                    scale_min=0.0,
                    scale_max=10.0,
                ),
                RubricCriterion(
                    name="coherence",
                    description="How coherent is the text",
                    weight=1.5,
                    scale_min=0.0,
                    scale_max=10.0,
                ),
            ],
        )

    def test_create_rubric_score_stage_defaults(self) -> None:
        """Test creating rubric score stage with defaults."""
        stage = create_rubric_score_stage()

        assert stage.name == "rubric_score"
        assert stage.stage_type == StageType.RUBRIC_SCORE
        assert stage.version == "1.0.0"
        assert stage.max_tokens == 512
        assert stage.temperature == 0.0
        assert stage.use_guided_decoding is True

    def test_create_rubric_score_stage_with_rubric(
        self, sample_rubric: Rubric
    ) -> None:
        """Test creating rubric score stage with rubric."""
        stage = create_rubric_score_stage(
            name="quality_scorer",
            rubric=sample_rubric,
            text_field="content",
        )

        assert stage.name == "quality_scorer"
        assert stage.rubric == sample_rubric
        assert stage.text_field == "content"

    def test_rubric_score_stage_input_schema(self) -> None:
        """Test rubric score stage input schema."""
        stage = create_rubric_score_stage(text_field="content")

        assert "content" in stage.input_schema
        assert "row_id" in stage.input_schema

    def test_rubric_score_stage_output_schema(self) -> None:
        """Test rubric score stage output schema."""
        stage = create_rubric_score_stage()

        assert "scores" in stage.output_schema
        assert "overall_score" in stage.output_schema
        assert "score_reasoning" in stage.output_schema
        assert "score_raw" in stage.output_schema

    def test_rubric_score_stage_fingerprint(self) -> None:
        """Test that stage fingerprint is stable."""
        stage1 = create_rubric_score_stage(name="test", max_tokens=512)
        stage2 = create_rubric_score_stage(name="test", max_tokens=512)
        stage3 = create_rubric_score_stage(name="test", max_tokens=256)

        assert stage1.fingerprint() == stage2.fingerprint()
        assert stage1.fingerprint() != stage3.fingerprint()

    def test_build_rubric_prompt_without_rubric(self) -> None:
        """Test rubric prompt building without rubric."""
        stage = create_rubric_score_stage()
        prompt = stage._build_rubric_prompt("Sample text to score")

        assert "Sample text to score" in prompt
        assert "Score the following content" in prompt

    def test_build_rubric_prompt_with_rubric(self, sample_rubric: Rubric) -> None:
        """Test rubric prompt building with rubric."""
        stage = create_rubric_score_stage(rubric=sample_rubric)
        prompt = stage._build_rubric_prompt("Sample text to score")

        assert "Sample text to score" in prompt
        assert "text_quality" in prompt
        assert "clarity" in prompt
        assert "coherence" in prompt
        assert "JSON" in prompt

    def test_get_output_schema_without_rubric(self) -> None:
        """Test output schema generation without rubric."""
        stage = create_rubric_score_stage()
        schema = stage._get_output_schema()

        assert schema["type"] == "object"
        assert "scores" in schema["properties"]
        assert "overall_score" in schema["properties"]
        assert "reasoning" in schema["properties"]

    def test_get_output_schema_with_rubric(self, sample_rubric: Rubric) -> None:
        """Test output schema generation with rubric."""
        stage = create_rubric_score_stage(rubric=sample_rubric)
        schema = stage._get_output_schema()

        assert schema["type"] == "object"
        scores_schema = schema["properties"]["scores"]
        assert "clarity" in scores_schema["properties"]
        assert "coherence" in scores_schema["properties"]

    def test_rubric_score_preprocess_fn(self, sample_rubric: Rubric) -> None:
        """Test preprocessing function creation."""
        stage = create_rubric_score_stage(rubric=sample_rubric)

        context = StageContext(
            run_id="test_run",
            stage_index=0,
            output_dir="/tmp/output",
        )

        preprocess_fn = stage._create_preprocess_fn(context)

        # Test with sample row
        row = {"row_id": "row_123", "text": "Sample text"}
        result = preprocess_fn(row)

        assert "messages" in result
        assert "sampling_params" in result
        assert result["row_id"] == "row_123"
        assert "guided_decoding" in result["sampling_params"]

    def test_rubric_score_postprocess_fn_valid_json(
        self, sample_rubric: Rubric
    ) -> None:
        """Test postprocessing with valid JSON output."""
        stage = create_rubric_score_stage(rubric=sample_rubric)
        postprocess_fn = stage._create_postprocess_fn()

        row = {
            "row_id": "row_123",
            "generated_text": '{"scores": {"clarity": 8.0, "coherence": 7.5}, "overall_score": 7.7, "reasoning": "Good quality text."}',
        }
        result = postprocess_fn(row)

        assert result["row_id"] == "row_123"
        assert result["scores"] == {"clarity": 8.0, "coherence": 7.5}
        assert result["overall_score"] == 7.7
        assert result["score_reasoning"] == "Good quality text."
        assert "parse_error" not in result

    def test_rubric_score_postprocess_fn_invalid_json(self) -> None:
        """Test postprocessing with invalid JSON output."""
        stage = create_rubric_score_stage()
        postprocess_fn = stage._create_postprocess_fn()

        row = {
            "row_id": "row_123",
            "generated_text": "This is not valid JSON",
        }
        result = postprocess_fn(row)

        assert result["row_id"] == "row_123"
        assert result["parse_error"] is True
        assert result["scores"] == {}
        assert result["overall_score"] == 0.0


class TestStageConfiguration:
    """Tests for stage configuration and fingerprinting."""

    def test_stage_config_fingerprint(self) -> None:
        """Test StageConfig fingerprint computation."""
        config1 = StageConfig(
            name="test",
            stage_type=StageType.VLM_CAPTION,
            version="1.0.0",
            params={"max_tokens": 256},
        )
        config2 = StageConfig(
            name="test",
            stage_type=StageType.VLM_CAPTION,
            version="1.0.0",
            params={"max_tokens": 256},
        )
        config3 = StageConfig(
            name="test",
            stage_type=StageType.VLM_CAPTION,
            version="1.0.1",
            params={"max_tokens": 256},
        )

        assert config1.fingerprint() == config2.fingerprint()
        assert config1.fingerprint() != config3.fingerprint()

    def test_stage_context_output_path(self) -> None:
        """Test StageContext output path generation."""
        context = StageContext(
            run_id="run_123",
            stage_index=0,
            output_dir="/output/run_123/stage_0",
        )

        assert context.get_output_path() == "/output/run_123/stage_0"
        assert (
            context.get_output_path("data.parquet")
            == "/output/run_123/stage_0/data.parquet"
        )


class TestStageIntegration:
    """Integration tests for stage components."""

    def test_vlm_and_rubric_stages_different_types(self) -> None:
        """Test that VLM and rubric stages have different types."""
        vlm_stage = create_vlm_caption_stage()
        rubric_stage = create_rubric_score_stage()

        assert vlm_stage.stage_type == StageType.VLM_CAPTION
        assert rubric_stage.stage_type == StageType.RUBRIC_SCORE
        assert vlm_stage.stage_type != rubric_stage.stage_type

    def test_stage_fingerprints_differ_across_types(self) -> None:
        """Test that fingerprints differ across stage types."""
        vlm_stage = create_vlm_caption_stage(name="test")
        rubric_stage = create_rubric_score_stage(name="test")

        # Same name but different types should have different fingerprints
        assert vlm_stage.fingerprint() != rubric_stage.fingerprint()

    def test_stages_inherit_from_base(self) -> None:
        """Test that concrete stages inherit from Stage base class."""
        from labelforge.pipelines.stage import Stage

        vlm_stage = create_vlm_caption_stage()
        rubric_stage = create_rubric_score_stage()

        assert isinstance(vlm_stage, Stage)
        assert isinstance(rubric_stage, Stage)
