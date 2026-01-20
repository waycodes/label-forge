"""Tests for prompt pack and fingerprinting."""

import pytest
import tempfile
from pathlib import Path

from labelforge.core.prompt_pack import (
    PromptTemplate,
    Rubric,
    RubricCriterion,
    OutputSchema,
    FewShotExample,
    PromptPack,
)
from labelforge.core.prompt_fingerprint import (
    compute_prompt_hash,
    compute_template_hash,
    compute_rubric_hash,
    compute_pack_hash,
    PromptFingerprint,
)


class TestPromptTemplate:
    """Tests for PromptTemplate model."""

    def test_create_template(self):
        """Template should be creatable."""
        template = PromptTemplate(
            name="caption",
            description="Basic captioning prompt",
            template="Describe this image: {text}",
        )
        assert template.name == "caption"
        assert "{text}" in template.template

    def test_template_with_variables(self):
        """Template with required variables should work."""
        template = PromptTemplate(
            name="caption",
            description="With context",
            template="Context: {context}\nDescribe: {text}",
            required_vars=["context"],
            optional_vars=["style"],
        )
        assert "context" in template.required_vars

    def test_template_version(self):
        """Template should have version."""
        template = PromptTemplate(
            name="caption",
            description="Test",
            template="Test",
            version="2.0.0",
        )
        assert template.version == "2.0.0"


class TestRubricCriterion:
    """Tests for RubricCriterion model."""

    def test_create_criterion(self):
        """Criterion should be creatable."""
        criterion = RubricCriterion(
            name="accuracy",
            description="How accurately the caption describes the image",
            weight=1.0,
            scale_min=0.0,
            scale_max=10.0,
        )
        assert criterion.name == "accuracy"
        assert criterion.weight == 1.0

    def test_criterion_with_levels(self):
        """Criterion with level descriptions should work."""
        criterion = RubricCriterion(
            name="accuracy",
            description="Accuracy of description",
            weight=1.0,
            scale_min=0.0,
            scale_max=10.0,
            levels={
                "0-3": "Poor",
                "4-6": "Fair",
                "7-9": "Good",
                "10": "Excellent",
            },
        )
        assert criterion.levels["7-9"] == "Good"


class TestRubric:
    """Tests for Rubric model."""

    def test_create_rubric(self):
        """Rubric should be creatable."""
        rubric = Rubric(
            name="caption_quality",
            version="1.0.0",
            description="Evaluates caption quality",
            criteria=[
                RubricCriterion(
                    name="accuracy",
                    description="Accuracy",
                    weight=1.0,
                ),
            ],
        )
        assert rubric.name == "caption_quality"
        assert len(rubric.criteria) == 1

    def test_rubric_with_multiple_criteria(self):
        """Rubric with multiple criteria should work."""
        rubric = Rubric(
            name="comprehensive",
            version="1.0.0",
            description="Comprehensive evaluation",
            criteria=[
                RubricCriterion(name="accuracy", description="", weight=1.0),
                RubricCriterion(name="detail", description="", weight=0.8),
                RubricCriterion(name="fluency", description="", weight=0.5),
            ],
            overall_scale_min=0.0,
            overall_scale_max=10.0,
        )
        assert len(rubric.criteria) == 3
        assert rubric.overall_scale_max == 10.0


class TestOutputSchema:
    """Tests for OutputSchema model."""

    def test_create_json_schema(self):
        """JSON schema should be creatable."""
        schema = OutputSchema(
            schema_type="json",
            schema_def={
                "type": "object",
                "properties": {
                    "caption": {"type": "string"},
                },
                "required": ["caption"],
            },
        )
        assert schema.schema_type == "json"
        assert "properties" in schema.schema_def

    def test_strict_schema(self):
        """Strict schema should be settable."""
        schema = OutputSchema(
            schema_type="json",
            schema_def={"type": "object"},
            strict=True,
        )
        assert schema.strict is True


class TestFewShotExample:
    """Tests for FewShotExample model."""

    def test_create_example(self):
        """Example should be creatable."""
        example = FewShotExample(
            input="[Image of a cat]",
            output="A fluffy orange tabby cat sleeping on a blue cushion.",
        )
        assert "cat" in example.input
        assert "orange tabby" in example.output

    def test_example_with_explanation(self):
        """Example with explanation should work."""
        example = FewShotExample(
            input="Test input",
            output="Test output",
            explanation="This demonstrates the correct format",
        )
        assert "demonstrates" in example.explanation


class TestPromptPack:
    """Tests for PromptPack model."""

    @pytest.fixture
    def sample_pack(self):
        """Create a sample prompt pack."""
        return PromptPack(
            name="test_pack",
            version="1.0.0",
            description="Test pack",
            templates={
                "caption": PromptTemplate(
                    name="caption",
                    description="Caption prompt",
                    template="Describe this image.",
                ),
            },
            rubrics={
                "quality": Rubric(
                    name="quality",
                    version="1.0.0",
                    description="Quality rubric",
                    criteria=[
                        RubricCriterion(
                            name="accuracy",
                            description="Accuracy",
                            weight=1.0,
                        ),
                    ],
                ),
            },
        )

    def test_create_pack(self, sample_pack):
        """Pack should be creatable."""
        assert sample_pack.name == "test_pack"
        assert "caption" in sample_pack.templates

    def test_pack_hash_stability(self, sample_pack):
        """Same pack should have same hash."""
        hash1 = compute_pack_hash(sample_pack)
        hash2 = compute_pack_hash(sample_pack)
        assert hash1 == hash2

    def test_pack_serialization(self, sample_pack):
        """Pack should be serializable."""
        data = sample_pack.model_dump()
        recovered = PromptPack.model_validate(data)
        assert recovered.name == sample_pack.name


class TestPromptFingerprint:
    """Tests for prompt fingerprinting."""

    def test_template_hash_stability(self):
        """Same template should produce same hash."""
        template = PromptTemplate(
            name="test",
            description="Test",
            template="Test template",
        )
        hash1 = compute_template_hash(template)
        hash2 = compute_template_hash(template)
        assert hash1 == hash2

    def test_template_hash_different(self):
        """Different templates should produce different hashes."""
        t1 = PromptTemplate(name="test", description="", template="Template 1")
        t2 = PromptTemplate(name="test", description="", template="Template 2")
        assert compute_template_hash(t1) != compute_template_hash(t2)

    def test_rubric_hash_stability(self):
        """Same rubric should produce same hash."""
        rubric = Rubric(
            name="test",
            version="1.0.0",
            description="Test",
            criteria=[
                RubricCriterion(name="c1", description="", weight=1.0),
            ],
        )
        hash1 = compute_rubric_hash(rubric)
        hash2 = compute_rubric_hash(rubric)
        assert hash1 == hash2

    def test_prompt_hash_stability(self):
        """Same prompt should produce same hash."""
        prompt = "Describe this image in detail."
        system = "You are a helpful assistant."
        
        hash1 = compute_prompt_hash(prompt, system)
        hash2 = compute_prompt_hash(prompt, system)
        assert hash1 == hash2

    def test_prompt_hash_without_system(self):
        """Prompt without system should work."""
        prompt = "Test prompt"
        hash1 = compute_prompt_hash(prompt, None)
        hash2 = compute_prompt_hash(prompt, None)
        assert hash1 == hash2

    def test_fingerprint_class(self):
        """PromptFingerprint class should work."""
        template = PromptTemplate(
            name="test",
            description="",
            template="Test",
        )
        fp = PromptFingerprint.from_template(template)
        assert fp.template_hash is not None
        assert fp.combined_hash is not None
