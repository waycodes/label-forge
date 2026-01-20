"""Tests for core schema models."""

import pytest
from datetime import datetime

from labelforge.core.schema import (
    DataSource,
    DataSourceType,
    RowMetadata,
    MultimodalRow,
    StageOutput,
    CaptionOutput,
    TagOutput,
    RubricScore,
    EmbeddingOutput,
)


class TestDataSource:
    """Tests for DataSource model."""

    def test_create_file_source(self):
        """File source should be creatable."""
        source = DataSource(
            source_type=DataSourceType.FILE,
            source_uri="file:///path/to/image.jpg",
            source_key="image.jpg",
        )
        assert source.source_type == DataSourceType.FILE
        assert source.source_uri == "file:///path/to/image.jpg"

    def test_create_dataset_source(self):
        """Dataset source should be creatable."""
        source = DataSource(
            source_type=DataSourceType.DATASET,
            source_uri="hf://dataset/split/row",
            source_key="row_123",
        )
        assert source.source_type == DataSourceType.DATASET

    def test_create_url_source(self):
        """URL source should be creatable."""
        source = DataSource(
            source_type=DataSourceType.URL,
            source_uri="https://example.com/image.jpg",
        )
        assert source.source_type == DataSourceType.URL

    def test_frozen(self):
        """DataSource should be immutable."""
        source = DataSource(
            source_type=DataSourceType.FILE,
            source_uri="file:///path/to/image.jpg",
        )
        with pytest.raises(Exception):
            source.source_uri = "different"


class TestRowMetadata:
    """Tests for RowMetadata model."""

    def test_create_metadata(self):
        """Metadata should be creatable."""
        meta = RowMetadata(
            labels={"category": "cat"},
            scores={"quality": 0.95},
        )
        assert meta.labels["category"] == "cat"
        assert meta.scores["quality"] == 0.95

    def test_default_values(self):
        """Defaults should be set correctly."""
        meta = RowMetadata()
        assert meta.labels == {}
        assert meta.annotations == {}
        assert meta.scores == {}
        assert meta.custom == {}


class TestMultimodalRow:
    """Tests for MultimodalRow model."""

    def test_create_row(self):
        """Row should be creatable with required fields."""
        source = DataSource(
            source_type=DataSourceType.FILE,
            source_uri="file:///path/to/image.jpg",
        )
        row = MultimodalRow(
            row_id="lf_1234567890abcdef",
            source=source,
        )
        assert row.row_id == "lf_1234567890abcdef"
        assert row.source.source_uri == "file:///path/to/image.jpg"

    def test_create_row_with_text(self):
        """Row with text should work."""
        source = DataSource(
            source_type=DataSourceType.FILE,
            source_uri="file:///path/to/image.jpg",
        )
        row = MultimodalRow(
            row_id="lf_1234567890abcdef",
            source=source,
            text="A cat sitting on a mat",
        )
        assert row.text == "A cat sitting on a mat"

    def test_create_row_with_metadata(self):
        """Row with metadata should work."""
        source = DataSource(
            source_type=DataSourceType.FILE,
            source_uri="file:///path/to/image.jpg",
        )
        row = MultimodalRow(
            row_id="lf_1234567890abcdef",
            source=source,
            metadata=RowMetadata(custom={"custom_field": "custom_value"}),
        )
        assert row.metadata.custom["custom_field"] == "custom_value"

    def test_row_id_format(self):
        """Row ID should have correct format."""
        source = DataSource(
            source_type=DataSourceType.FILE,
            source_uri="file:///path/to/image.jpg",
        )
        row = MultimodalRow(
            row_id="lf_1234567890abcdef",
            source=source,
        )
        assert row.row_id.startswith("lf_")


class TestStageOutput:
    """Tests for StageOutput model."""

    def test_create_stage_output(self):
        """Stage output should be creatable."""
        output = StageOutput(
            row_id="lf_1234567890abcdef",
            stage_name="caption",
            stage_version="1.0.0",
            output={"caption": "A cat"},
            prompt_hash="abc123",
            model_hash="def456",
            sampling_params_hash="ghi789",
            latency_ms=150.5,
        )
        assert output.row_id == "lf_1234567890abcdef"
        assert output.stage_name == "caption"
        assert output.output["caption"] == "A cat"

    def test_create_output_with_error(self):
        """Output with error should work."""
        output = StageOutput(
            row_id="lf_1234567890abcdef",
            stage_name="caption",
            stage_version="1.0.0",
            output={},
            prompt_hash="abc123",
            model_hash="def456",
            sampling_params_hash="ghi789",
            latency_ms=50.0,
            error="Processing failed",
        )
        assert output.error == "Processing failed"


class TestCaptionOutput:
    """Tests for CaptionOutput model."""

    def test_create_caption_output(self):
        """Caption output should be creatable."""
        output = CaptionOutput(
            caption="A fluffy cat sitting on a red mat.",
        )
        assert output.caption == "A fluffy cat sitting on a red mat."

    def test_caption_with_confidence(self):
        """Caption with confidence should work."""
        output = CaptionOutput(
            caption="A cat",
            confidence=0.95,
        )
        assert output.confidence == 0.95


class TestTagOutput:
    """Tests for TagOutput model."""

    def test_create_tag_output(self):
        """Tag output should be creatable."""
        output = TagOutput(
            tags=["cat", "animal", "pet"],
        )
        assert output.tags == ["cat", "animal", "pet"]

    def test_tags_with_confidences(self):
        """Tags with confidences should work."""
        output = TagOutput(
            tags=["cat", "animal"],
            confidences={"cat": 0.95, "animal": 0.90},
        )
        assert output.confidences["cat"] == 0.95


class TestRubricScore:
    """Tests for RubricScore model."""

    def test_create_rubric_score(self):
        """Rubric score should be creatable."""
        score = RubricScore(
            score=8.5,
            explanation="Good caption with detailed description.",
        )
        assert score.score == 8.5

    def test_rubric_with_subscores(self):
        """Rubric with subscores should work."""
        score = RubricScore(
            score=8.0,
            subscores={"accuracy": 9.0, "detail": 7.5, "fluency": 8.0},
            explanation="Detailed breakdown.",
        )
        assert score.subscores["accuracy"] == 9.0

    def test_normalized_score(self):
        """Normalized score should be valid."""
        score = RubricScore(
            score=8.0,
            normalized_score=0.8,
            explanation="Test",
        )
        assert score.normalized_score == 0.8


class TestEmbeddingOutput:
    """Tests for EmbeddingOutput model."""

    def test_create_embedding_output(self):
        """Embedding output should be creatable."""
        output = EmbeddingOutput(
            embedding=[0.1, 0.2, 0.3, 0.4],
            embedding_model_hash="abc123",
            embedding_dim=4,
        )
        assert len(output.embedding) == 4
        assert output.embedding_dim == 4

    def test_embedding_dimensionality(self):
        """Embedding can have various dimensions."""
        output = EmbeddingOutput(
            embedding=[0.0] * 768,
            embedding_model_hash="bert_hash",
            embedding_dim=768,
        )
        assert len(output.embedding) == 768
