"""
Pipeline DAG schema.

Defines multi-stage workflows with dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from labelforge.core.json_canonical import canonical_json_bytes
import xxhash


@dataclass
class DAGNode:
    """A node in the pipeline DAG."""

    name: str
    stage_type: str
    config: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)

    def fingerprint(self) -> str:
        """Compute node fingerprint."""
        content = {
            "name": self.name,
            "stage_type": self.stage_type,
            "config": self.config,
            "depends_on": sorted(self.depends_on),
        }
        json_bytes = canonical_json_bytes(content)
        return xxhash.xxh64(json_bytes).hexdigest()


@dataclass
class PipelineDAG:
    """
    Directed Acyclic Graph of pipeline stages.

    Supports linear pipelines, branches, and merges.
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    nodes: list[DAGNode] = field(default_factory=list)

    def add_node(
        self,
        name: str,
        stage_type: str,
        config: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
    ) -> DAGNode:
        """
        Add a node to the DAG.

        Args:
            name: Node name (must be unique).
            stage_type: Type of stage.
            config: Stage configuration.
            depends_on: Names of upstream nodes.

        Returns:
            Created DAGNode.

        Raises:
            ValueError: If node name already exists.
        """
        if any(n.name == name for n in self.nodes):
            raise ValueError(f"Node '{name}' already exists")

        if depends_on:
            for dep in depends_on:
                if not any(n.name == dep for n in self.nodes):
                    raise ValueError(f"Dependency '{dep}' not found")

        node = DAGNode(
            name=name,
            stage_type=stage_type,
            config=config or {},
            depends_on=depends_on or [],
        )
        self.nodes.append(node)
        return node

    def get_node(self, name: str) -> DAGNode | None:
        """Get node by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_execution_order(self) -> list[str]:
        """
        Get topologically sorted execution order.

        Returns:
            List of node names in execution order.

        Raises:
            ValueError: If DAG has cycles.
        """
        # Kahn's algorithm for topological sort
        in_degree = {n.name: len(n.depends_on) for n in self.nodes}
        queue = [n.name for n in self.nodes if in_degree[n.name] == 0]
        result = []

        while queue:
            node_name = queue.pop(0)
            result.append(node_name)

            for n in self.nodes:
                if node_name in n.depends_on:
                    in_degree[n.name] -= 1
                    if in_degree[n.name] == 0:
                        queue.append(n.name)

        if len(result) != len(self.nodes):
            raise ValueError("DAG contains cycles")

        return result

    def get_upstream(self, name: str) -> list[str]:
        """Get all upstream nodes (transitive dependencies)."""
        node = self.get_node(name)
        if node is None:
            return []

        upstream = set()
        queue = list(node.depends_on)

        while queue:
            dep_name = queue.pop(0)
            if dep_name not in upstream:
                upstream.add(dep_name)
                dep_node = self.get_node(dep_name)
                if dep_node:
                    queue.extend(dep_node.depends_on)

        return list(upstream)

    def get_downstream(self, name: str) -> list[str]:
        """Get all downstream nodes (dependent on this node)."""
        downstream = set()

        for node in self.nodes:
            if name in node.depends_on:
                downstream.add(node.name)
                downstream.update(self.get_downstream(node.name))

        return list(downstream)

    def fingerprint(self) -> str:
        """Compute DAG fingerprint."""
        content = {
            "name": self.name,
            "version": self.version,
            "nodes": [
                {
                    "name": n.name,
                    "stage_type": n.stage_type,
                    "config": n.config,
                    "depends_on": sorted(n.depends_on),
                }
                for n in sorted(self.nodes, key=lambda x: x.name)
            ],
        }
        json_bytes = canonical_json_bytes(content)
        return xxhash.xxh64(json_bytes).hexdigest()

    def validate(self) -> list[str]:
        """
        Validate DAG structure.

        Returns:
            List of validation error messages.
        """
        errors = []

        # Check for unique names
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            errors.append("Duplicate node names found")

        # Check dependencies exist
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in names:
                    errors.append(f"Node '{node.name}' depends on unknown node '{dep}'")

        # Check for cycles
        try:
            self.get_execution_order()
        except ValueError:
            errors.append("DAG contains cycles")

        return errors


def create_linear_pipeline(
    name: str,
    stages: list[tuple[str, str, dict[str, Any]]],
) -> PipelineDAG:
    """
    Create a simple linear pipeline.

    Args:
        name: Pipeline name.
        stages: List of (name, stage_type, config) tuples.

    Returns:
        PipelineDAG with linear dependencies.

    Example:
        >>> dag = create_linear_pipeline("simple", [
        ...     ("caption", "vlm_caption", {}),
        ...     ("score", "rubric_score", {}),
        ... ])
        >>> dag.get_execution_order()
        ['caption', 'score']
    """
    dag = PipelineDAG(name=name)
    prev_name = None

    for stage_name, stage_type, config in stages:
        depends_on = [prev_name] if prev_name else []
        dag.add_node(stage_name, stage_type, config, depends_on)
        prev_name = stage_name

    return dag
