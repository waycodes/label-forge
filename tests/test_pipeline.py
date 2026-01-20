"""Tests for pipeline stage and runner."""

import pytest

from labelforge.pipelines.stage import (
    Stage,
    StageConfig,
    StageType,
    StageContext,
    StageResult,
)
from labelforge.pipelines.dag import (
    PipelineDAG,
    DAGNode,
    create_linear_pipeline,
)
from labelforge.pipelines.runner import (
    PipelineRunner,
    RunConfig,
    create_runner,
)


class TestStageConfig:
    """Tests for StageConfig."""

    def test_create_config(self):
        """Config should be creatable."""
        config = StageConfig(
            name="caption",
            stage_type=StageType.VLM_CAPTION,
        )
        assert config.name == "caption"
        assert config.stage_type == StageType.VLM_CAPTION

    def test_config_with_prompt(self):
        """Config with prompt should work."""
        config = StageConfig(
            name="caption",
            stage_type=StageType.VLM_CAPTION,
            prompt_pack="mvp",
            template_name="caption_basic",
        )
        assert config.prompt_pack == "mvp"

    def test_config_fingerprint_stability(self):
        """Same config should have same fingerprint."""
        config = StageConfig(
            name="caption",
            stage_type=StageType.VLM_CAPTION,
            version="1.0.0",
        )
        fp1 = config.fingerprint()
        fp2 = config.fingerprint()
        assert fp1 == fp2

    def test_config_fingerprint_different(self):
        """Different configs should have different fingerprints."""
        c1 = StageConfig(name="caption", stage_type=StageType.VLM_CAPTION)
        c2 = StageConfig(name="score", stage_type=StageType.RUBRIC_SCORE)
        assert c1.fingerprint() != c2.fingerprint()


class TestStageResult:
    """Tests for StageResult."""

    def test_create_success_result(self):
        """Success result should be creatable."""
        result = StageResult(
            success=True,
            output_path="/tmp/output",
            output_row_count=100,
            cache_hits=80,
            cache_misses=20,
        )
        assert result.success is True
        assert result.cache_hits == 80

    def test_create_error_result(self):
        """Error result should be creatable."""
        result = StageResult(
            success=False,
            error="Processing failed",
        )
        assert result.success is False
        assert result.error == "Processing failed"


class TestStageContext:
    """Tests for StageContext."""

    def test_create_context(self):
        """Context should be creatable."""
        context = StageContext(
            run_id="abc123",
            stage_index=0,
            output_dir="/tmp/output",
        )
        assert context.run_id == "abc123"
        assert context.cache_enabled is True  # default

    def test_context_with_fingerprints(self):
        """Context with fingerprints should work."""
        context = StageContext(
            run_id="abc123",
            stage_index=0,
            output_dir="/tmp/output",
            prompt_hash="abc",
            model_hash="def",
            sampling_params_hash="ghi",
        )
        assert context.prompt_hash == "abc"

    def test_get_output_path(self):
        """Should construct output path."""
        context = StageContext(
            run_id="abc123",
            stage_index=0,
            output_dir="/tmp/output",
        )
        path = context.get_output_path("data.parquet")
        assert path == "/tmp/output/data.parquet"


class TestStageTypes:
    """Tests for stage type enumeration."""

    def test_all_stage_types_exist(self):
        """All expected stage types should exist."""
        assert StageType.VLM_CAPTION is not None
        assert StageType.VLM_TAGS is not None
        assert StageType.TEXT_LABEL is not None
        assert StageType.RUBRIC_SCORE is not None
        assert StageType.EMBED_TEXT is not None
        assert StageType.RERANK is not None
        assert StageType.SYNTH_TEXT is not None
        assert StageType.FILTER is not None
        assert StageType.TRANSFORM is not None


class TestRunConfig:
    """Tests for RunConfig."""

    def test_create_default_config(self):
        """Default config should be creatable."""
        config = RunConfig()
        assert config.seed == 42
        assert config.cache_enabled is True
        assert config.deterministic_mode is True

    def test_create_custom_config(self):
        """Custom config should work."""
        config = RunConfig(
            run_name="test_run",
            output_dir="/custom/output",
            seed=123,
            cache_enabled=False,
        )
        assert config.run_name == "test_run"
        assert config.seed == 123
        assert config.cache_enabled is False


class TestPipelineRunner:
    """Tests for PipelineRunner."""

    @pytest.fixture
    def simple_dag(self):
        """Create a simple DAG for testing."""
        return create_linear_pipeline("test", [
            ("stage1", "vlm_caption", {}),
            ("stage2", "rubric_score", {}),
        ])

    def test_create_runner(self, simple_dag):
        """Runner should be creatable."""
        runner = create_runner(simple_dag)
        assert runner.dag == simple_dag

    def test_create_runner_with_config(self, simple_dag):
        """Runner with config should work."""
        config = RunConfig(seed=123)
        runner = create_runner(simple_dag, config)
        assert runner.config.seed == 123

    def test_runner_has_unique_run_id(self, simple_dag):
        """Each runner should have unique run ID."""
        runner1 = create_runner(simple_dag)
        runner2 = create_runner(simple_dag)
        assert runner1.run_id != runner2.run_id

    def test_get_run_dir(self, simple_dag):
        """Should construct run directory path."""
        config = RunConfig(output_dir="/tmp/runs")
        runner = PipelineRunner(dag=simple_dag, config=config)
        
        run_dir = runner.get_run_dir()
        assert str(run_dir).startswith("/tmp/runs")
        assert runner.run_id in str(run_dir)


class TestDAGIntegration:
    """Integration tests for DAG with runner."""

    def test_complex_dag(self):
        """Complex DAG should work."""
        dag = PipelineDAG(name="complex")
        
        # Input stage
        dag.add_node("input", "transform", {})
        
        # Parallel branches
        dag.add_node("caption", "vlm_caption", {}, depends_on=["input"])
        dag.add_node("tags", "vlm_tags", {}, depends_on=["input"])
        
        # Merge
        dag.add_node("score", "rubric_score", {}, depends_on=["caption", "tags"])
        
        # Final
        dag.add_node("filter", "filter", {}, depends_on=["score"])
        
        order = dag.get_execution_order()
        
        # Verify ordering constraints
        assert order.index("input") < order.index("caption")
        assert order.index("input") < order.index("tags")
        assert order.index("caption") < order.index("score")
        assert order.index("tags") < order.index("score")
        assert order.index("score") < order.index("filter")
