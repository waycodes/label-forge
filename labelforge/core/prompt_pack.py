"""
Prompt pack format and loading.

Prompts are first-class, versioned assets that include templates,
system instructions, rubrics, output schemas, and few-shot examples.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from labelforge.core.json_canonical import canonical_json_bytes


class FewShotExample(BaseModel):
    """A single few-shot example for in-context learning."""

    model_config = ConfigDict(frozen=True)

    input: str = Field(description="Example input text or description")
    output: str = Field(description="Expected output")
    explanation: str | None = Field(
        default=None, description="Optional explanation of why this output is correct"
    )


class OutputSchema(BaseModel):
    """JSON schema for structured output."""

    model_config = ConfigDict(frozen=True)

    schema_type: str = Field(default="json", description="Schema type: json, pydantic")
    schema_def: dict[str, Any] = Field(description="JSON Schema definition")
    strict: bool = Field(
        default=True, description="Whether to enforce strict schema adherence"
    )


class RubricCriterion(BaseModel):
    """A single criterion in a scoring rubric."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Criterion name")
    description: str = Field(description="What this criterion measures")
    weight: float = Field(default=1.0, ge=0.0, description="Weight in overall score")
    scale_min: float = Field(default=0.0, description="Minimum score value")
    scale_max: float = Field(default=10.0, description="Maximum score value")
    levels: dict[str, str] | None = Field(
        default=None, description="Description of each score level"
    )


class Rubric(BaseModel):
    """A complete scoring rubric."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Rubric name")
    version: str = Field(description="Rubric version")
    description: str = Field(description="What this rubric evaluates")
    criteria: list[RubricCriterion] = Field(description="Scoring criteria")
    overall_scale_min: float = Field(default=0.0, description="Overall min score")
    overall_scale_max: float = Field(default=10.0, description="Overall max score")
    output_schema: OutputSchema | None = Field(
        default=None, description="Schema for structured score output"
    )


class PromptTemplate(BaseModel):
    """A single prompt template."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Template name")
    template: str = Field(description="Prompt template with {placeholders}")
    description: str | None = Field(default=None, description="Template description")
    required_vars: list[str] = Field(
        default_factory=list, description="Required placeholder variables"
    )
    optional_vars: list[str] = Field(
        default_factory=list, description="Optional placeholder variables"
    )


class PromptPack(BaseModel):
    """
    A versioned collection of prompts and related assets.

    Prompt packs are the unit of prompt versioning in LabelForge.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    name: str = Field(description="Pack name")
    version: str = Field(description="Semantic version (e.g., 1.0.0)")
    description: str = Field(description="Pack description")

    # Content
    system_instruction: str | None = Field(
        default=None, description="Default system instruction"
    )
    templates: dict[str, PromptTemplate] = Field(
        default_factory=dict, description="Named prompt templates"
    )
    rubrics: dict[str, Rubric] = Field(
        default_factory=dict, description="Named scoring rubrics"
    )
    output_schemas: dict[str, OutputSchema] = Field(
        default_factory=dict, description="Named output schemas"
    )
    few_shot_examples: dict[str, list[FewShotExample]] = Field(
        default_factory=dict, description="Named sets of few-shot examples"
    )

    # Metadata
    author: str | None = Field(default=None, description="Pack author")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    content_hash: str | None = Field(
        default=None, description="Content hash for fingerprinting"
    )

    @classmethod
    def load(cls, path: Path) -> PromptPack:
        """
        Load a prompt pack from a YAML file.

        Args:
            path: Path to pack.yaml file.

        Returns:
            Loaded PromptPack instance.
        """
        content = path.read_text()
        data = yaml.safe_load(content)
        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        """
        Save prompt pack to a YAML file.

        Args:
            path: Path to save to.
        """
        data = self.model_dump(exclude_none=True)
        content = yaml.dump(data, default_flow_style=False, sort_keys=True)
        path.write_text(content)

    def get_template(self, name: str) -> PromptTemplate:
        """
        Get a template by name.

        Args:
            name: Template name.

        Returns:
            The template.

        Raises:
            KeyError: If template not found.
        """
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found in pack '{self.name}'")
        return self.templates[name]

    def get_rubric(self, name: str) -> Rubric:
        """
        Get a rubric by name.

        Args:
            name: Rubric name.

        Returns:
            The rubric.

        Raises:
            KeyError: If rubric not found.
        """
        if name not in self.rubrics:
            raise KeyError(f"Rubric '{name}' not found in pack '{self.name}'")
        return self.rubrics[name]

    def get_output_schema(self, name: str) -> OutputSchema:
        """
        Get an output schema by name.

        Args:
            name: Schema name.

        Returns:
            The schema.

        Raises:
            KeyError: If schema not found.
        """
        if name not in self.output_schemas:
            raise KeyError(f"Output schema '{name}' not found in pack '{self.name}'")
        return self.output_schemas[name]

    def get_examples(self, name: str) -> list[FewShotExample]:
        """
        Get few-shot examples by name.

        Args:
            name: Example set name.

        Returns:
            List of examples.

        Raises:
            KeyError: If example set not found.
        """
        if name not in self.few_shot_examples:
            raise KeyError(f"Example set '{name}' not found in pack '{self.name}'")
        return self.few_shot_examples[name]

    def compute_hash(self) -> str:
        """
        Compute content hash for fingerprinting.

        Returns:
            Hex-encoded hash of pack contents.
        """
        import xxhash

        # Serialize content deterministically
        content = self.model_dump(exclude={"content_hash", "created_at"})
        json_bytes = canonical_json_bytes(content)
        return xxhash.xxh64(json_bytes).hexdigest()


def render_template(
    template: PromptTemplate,
    variables: dict[str, Any],
    strict: bool = True,
) -> str:
    """
    Render a prompt template with variables.

    Args:
        template: The template to render.
        variables: Variable values to substitute.
        strict: If True, raise on missing required variables.

    Returns:
        Rendered prompt string.

    Raises:
        ValueError: If strict and required variables are missing.
    """
    # Check required variables
    if strict:
        missing = set(template.required_vars) - set(variables.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

    # Build full variable dict with defaults for optional vars
    full_vars = {var: "" for var in template.optional_vars}
    full_vars.update(variables)

    # Render template
    return template.template.format(**full_vars)
