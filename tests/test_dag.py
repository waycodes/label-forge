"""Tests for pipeline DAG functionality."""

import pytest

from labelforge.pipelines.dag import PipelineDAG, DAGNode, create_linear_pipeline


class TestDAGNode:
    """Tests for DAG nodes."""

    def test_create_node(self):
        """Nodes should be creatable."""
        node = DAGNode(
            name="test",
            stage_type="vlm_caption",
            config={"param": "value"},
            depends_on=["upstream"],
        )
        assert node.name == "test"
        assert node.stage_type == "vlm_caption"

    def test_node_fingerprint_stability(self):
        """Same node should produce same fingerprint."""
        node1 = DAGNode(name="test", stage_type="vlm_caption")
        node2 = DAGNode(name="test", stage_type="vlm_caption")
        assert node1.fingerprint() == node2.fingerprint()

    def test_node_fingerprint_different(self):
        """Different nodes should produce different fingerprints."""
        node1 = DAGNode(name="test1", stage_type="vlm_caption")
        node2 = DAGNode(name="test2", stage_type="vlm_caption")
        assert node1.fingerprint() != node2.fingerprint()


class TestPipelineDAG:
    """Tests for pipeline DAG operations."""

    def test_create_empty_dag(self):
        """Empty DAG should be creatable."""
        dag = PipelineDAG(name="test")
        assert dag.name == "test"
        assert len(dag.nodes) == 0

    def test_add_node(self):
        """Nodes should be addable."""
        dag = PipelineDAG(name="test")
        node = dag.add_node("stage1", "vlm_caption")
        assert node.name == "stage1"
        assert len(dag.nodes) == 1

    def test_add_node_with_dependency(self):
        """Nodes with dependencies should be addable."""
        dag = PipelineDAG(name="test")
        dag.add_node("stage1", "vlm_caption")
        dag.add_node("stage2", "rubric_score", depends_on=["stage1"])
        
        assert len(dag.nodes) == 2
        assert dag.nodes[1].depends_on == ["stage1"]

    def test_add_duplicate_node_fails(self):
        """Adding duplicate node should fail."""
        dag = PipelineDAG(name="test")
        dag.add_node("stage1", "vlm_caption")
        
        with pytest.raises(ValueError, match="already exists"):
            dag.add_node("stage1", "vlm_caption")

    def test_add_node_missing_dependency_fails(self):
        """Adding node with missing dependency should fail."""
        dag = PipelineDAG(name="test")
        
        with pytest.raises(ValueError, match="not found"):
            dag.add_node("stage2", "rubric_score", depends_on=["stage1"])

    def test_execution_order_linear(self):
        """Linear DAG should have correct execution order."""
        dag = PipelineDAG(name="test")
        dag.add_node("stage1", "vlm_caption")
        dag.add_node("stage2", "rubric_score", depends_on=["stage1"])
        dag.add_node("stage3", "filter", depends_on=["stage2"])
        
        order = dag.get_execution_order()
        assert order == ["stage1", "stage2", "stage3"]

    def test_execution_order_diamond(self):
        """Diamond DAG should have valid execution order."""
        dag = PipelineDAG(name="test")
        dag.add_node("input", "transform")
        dag.add_node("branch_a", "vlm_caption", depends_on=["input"])
        dag.add_node("branch_b", "text_label", depends_on=["input"])
        dag.add_node("merge", "filter", depends_on=["branch_a", "branch_b"])
        
        order = dag.get_execution_order()
        
        # input must come first
        assert order.index("input") < order.index("branch_a")
        assert order.index("input") < order.index("branch_b")
        # merge must come last
        assert order.index("branch_a") < order.index("merge")
        assert order.index("branch_b") < order.index("merge")

    def test_get_node(self):
        """Should be able to get node by name."""
        dag = PipelineDAG(name="test")
        dag.add_node("stage1", "vlm_caption")
        
        node = dag.get_node("stage1")
        assert node is not None
        assert node.name == "stage1"
        
        assert dag.get_node("nonexistent") is None

    def test_get_upstream(self):
        """Should correctly identify upstream nodes."""
        dag = PipelineDAG(name="test")
        dag.add_node("stage1", "vlm_caption")
        dag.add_node("stage2", "rubric_score", depends_on=["stage1"])
        dag.add_node("stage3", "filter", depends_on=["stage2"])
        
        upstream = dag.get_upstream("stage3")
        assert set(upstream) == {"stage1", "stage2"}

    def test_get_downstream(self):
        """Should correctly identify downstream nodes."""
        dag = PipelineDAG(name="test")
        dag.add_node("stage1", "vlm_caption")
        dag.add_node("stage2", "rubric_score", depends_on=["stage1"])
        dag.add_node("stage3", "filter", depends_on=["stage2"])
        
        downstream = dag.get_downstream("stage1")
        assert set(downstream) == {"stage2", "stage3"}

    def test_validate_valid_dag(self):
        """Valid DAG should pass validation."""
        dag = PipelineDAG(name="test")
        dag.add_node("stage1", "vlm_caption")
        dag.add_node("stage2", "rubric_score", depends_on=["stage1"])
        
        errors = dag.validate()
        assert len(errors) == 0

    def test_fingerprint_stability(self):
        """Same DAG should produce same fingerprint."""
        dag1 = create_linear_pipeline("test", [
            ("stage1", "vlm_caption", {}),
            ("stage2", "rubric_score", {}),
        ])
        dag2 = create_linear_pipeline("test", [
            ("stage1", "vlm_caption", {}),
            ("stage2", "rubric_score", {}),
        ])
        assert dag1.fingerprint() == dag2.fingerprint()


class TestLinearPipeline:
    """Tests for linear pipeline helper."""

    def test_create_linear_pipeline(self):
        """Linear pipeline should be creatable."""
        dag = create_linear_pipeline("test", [
            ("caption", "vlm_caption", {"model": "llava"}),
            ("score", "rubric_score", {}),
            ("filter", "filter", {}),
        ])
        
        assert len(dag.nodes) == 3
        assert dag.nodes[0].depends_on == []
        assert dag.nodes[1].depends_on == ["caption"]
        assert dag.nodes[2].depends_on == ["score"]

    def test_linear_pipeline_execution_order(self):
        """Linear pipeline should execute in order."""
        dag = create_linear_pipeline("test", [
            ("a", "type1", {}),
            ("b", "type2", {}),
            ("c", "type3", {}),
        ])
        
        order = dag.get_execution_order()
        assert order == ["a", "b", "c"]
