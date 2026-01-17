"""
Prompt fingerprinting for cache keys and manifests.

Computes stable hashes of prompt content for reproducibility tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import xxhash

from labelforge.core.json_canonical import canonical_json_bytes

if TYPE_CHECKING:
    from labelforge.core.prompt_pack import PromptPack, PromptTemplate, Rubric


def compute_prompt_hash(
    template_text: str,
    system_instruction: str | None = None,
    output_schema: dict[str, Any] | None = None,
    few_shot_examples: list[dict[str, str]] | None = None,
) -> str:
    """
    Compute a hash of prompt components.

    The hash includes all components that affect model output:
    - Template text
    - System instruction
    - Output schema
    - Few-shot examples

    Args:
        template_text: The prompt template text.
        system_instruction: Optional system instruction.
        output_schema: Optional JSON schema for structured output.
        few_shot_examples: Optional list of few-shot examples.

    Returns:
        Hex-encoded hash string.

    Examples:
        >>> compute_prompt_hash("Describe this image")
        '7a8b9c0d1e2f3a4b'  # Example hash
    """
    components: dict[str, Any] = {
        "template": template_text,
    }

    if system_instruction:
        components["system"] = system_instruction

    if output_schema:
        components["schema"] = output_schema

    if few_shot_examples:
        components["examples"] = few_shot_examples

    json_bytes = canonical_json_bytes(components)
    return xxhash.xxh64(json_bytes).hexdigest()


def compute_template_hash(template: PromptTemplate) -> str:
    """
    Compute hash of a prompt template.

    Args:
        template: The template to hash.

    Returns:
        Hex-encoded hash string.
    """
    content = {
        "name": template.name,
        "template": template.template,
        "required_vars": template.required_vars,
        "optional_vars": template.optional_vars,
    }
    json_bytes = canonical_json_bytes(content)
    return xxhash.xxh64(json_bytes).hexdigest()


def compute_rubric_hash(rubric: Rubric) -> str:
    """
    Compute hash of a scoring rubric.

    Args:
        rubric: The rubric to hash.

    Returns:
        Hex-encoded hash string.
    """
    content = {
        "name": rubric.name,
        "version": rubric.version,
        "criteria": [
            {
                "name": c.name,
                "description": c.description,
                "weight": c.weight,
                "scale_min": c.scale_min,
                "scale_max": c.scale_max,
                "levels": c.levels,
            }
            for c in rubric.criteria
        ],
        "overall_scale_min": rubric.overall_scale_min,
        "overall_scale_max": rubric.overall_scale_max,
    }
    if rubric.output_schema:
        content["output_schema"] = rubric.output_schema.schema_def

    json_bytes = canonical_json_bytes(content)
    return xxhash.xxh64(json_bytes).hexdigest()


def compute_pack_hash(pack: PromptPack) -> str:
    """
    Compute hash of an entire prompt pack.

    Args:
        pack: The prompt pack to hash.

    Returns:
        Hex-encoded hash string.
    """
    return pack.compute_hash()


class PromptFingerprint:
    """
    Complete fingerprint for a prompt configuration.

    Includes all components that affect the model output.
    """

    def __init__(
        self,
        pack_name: str,
        pack_version: str,
        template_name: str,
        template_hash: str,
        system_hash: str | None = None,
        schema_hash: str | None = None,
        examples_hash: str | None = None,
    ):
        self.pack_name = pack_name
        self.pack_version = pack_version
        self.template_name = template_name
        self.template_hash = template_hash
        self.system_hash = system_hash
        self.schema_hash = schema_hash
        self.examples_hash = examples_hash

    @property
    def combined_hash(self) -> str:
        """Compute combined hash of all components."""
        components = [
            self.pack_name,
            self.pack_version,
            self.template_name,
            self.template_hash,
            self.system_hash or "",
            self.schema_hash or "",
            self.examples_hash or "",
        ]
        combined = "|".join(components)
        return xxhash.xxh64(combined.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pack_name": self.pack_name,
            "pack_version": self.pack_version,
            "template_name": self.template_name,
            "template_hash": self.template_hash,
            "system_hash": self.system_hash,
            "schema_hash": self.schema_hash,
            "examples_hash": self.examples_hash,
            "combined_hash": self.combined_hash,
        }

    @classmethod
    def from_pack(
        cls,
        pack: PromptPack,
        template_name: str,
        example_set_name: str | None = None,
        output_schema_name: str | None = None,
    ) -> PromptFingerprint:
        """
        Create fingerprint from a prompt pack.

        Args:
            pack: The prompt pack.
            template_name: Name of the template to use.
            example_set_name: Optional name of few-shot example set.
            output_schema_name: Optional name of output schema.

        Returns:
            PromptFingerprint instance.
        """
        template = pack.get_template(template_name)
        template_hash = compute_template_hash(template)

        system_hash = None
        if pack.system_instruction:
            system_hash = xxhash.xxh64(
                pack.system_instruction.encode("utf-8")
            ).hexdigest()

        schema_hash = None
        if output_schema_name:
            schema = pack.get_output_schema(output_schema_name)
            schema_bytes = canonical_json_bytes(schema.schema_def)
            schema_hash = xxhash.xxh64(schema_bytes).hexdigest()

        examples_hash = None
        if example_set_name:
            examples = pack.get_examples(example_set_name)
            examples_content = [
                {"input": e.input, "output": e.output} for e in examples
            ]
            examples_bytes = canonical_json_bytes(examples_content)
            examples_hash = xxhash.xxh64(examples_bytes).hexdigest()

        return cls(
            pack_name=pack.name,
            pack_version=pack.version,
            template_name=template_name,
            template_hash=template_hash,
            system_hash=system_hash,
            schema_hash=schema_hash,
            examples_hash=examples_hash,
        )
