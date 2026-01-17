"""
Prompt resolution from (name, version) to concrete prompt strings.

Loads prompt packs, validates them, and returns ready-to-use prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labelforge.core.prompt_fingerprint import PromptFingerprint
from labelforge.core.prompt_pack import (
    FewShotExample,
    OutputSchema,
    PromptPack,
    Rubric,
    render_template,
)


class PromptResolver:
    """
    Resolves prompt references to concrete prompt strings.

    Loads prompt packs from disk and provides access to templates,
    rubrics, and schemas with validation and fingerprinting.
    """

    def __init__(self, prompts_dir: Path):
        """
        Initialize resolver with prompts directory.

        Args:
            prompts_dir: Root directory containing prompt packs.
                Expected structure: prompts_dir/<pack_name>/pack.yaml
        """
        self.prompts_dir = prompts_dir
        self._cache: dict[str, PromptPack] = {}

    def load_pack(self, pack_name: str, version: str | None = None) -> PromptPack:
        """
        Load a prompt pack by name.

        Args:
            pack_name: Name of the pack directory.
            version: Optional version to validate against.

        Returns:
            Loaded PromptPack.

        Raises:
            FileNotFoundError: If pack not found.
            ValueError: If version mismatch.
        """
        cache_key = pack_name

        if cache_key not in self._cache:
            pack_path = self.prompts_dir / pack_name / "pack.yaml"

            if not pack_path.exists():
                raise FileNotFoundError(
                    f"Prompt pack '{pack_name}' not found at {pack_path}"
                )

            pack = PromptPack.load(pack_path)
            self._cache[cache_key] = pack

        pack = self._cache[cache_key]

        if version and pack.version != version:
            raise ValueError(
                f"Version mismatch for pack '{pack_name}': "
                f"requested {version}, found {pack.version}"
            )

        return pack

    def resolve_prompt(
        self,
        pack_name: str,
        template_name: str,
        variables: dict[str, Any],
        version: str | None = None,
        include_system: bool = True,
        example_set: str | None = None,
    ) -> ResolvedPrompt:
        """
        Resolve a prompt reference to concrete strings.

        Args:
            pack_name: Name of the prompt pack.
            template_name: Name of the template.
            variables: Variables to substitute in template.
            version: Optional version to validate.
            include_system: Whether to include system instruction.
            example_set: Optional name of few-shot example set.

        Returns:
            ResolvedPrompt with concrete strings and fingerprint.
        """
        pack = self.load_pack(pack_name, version)
        template = pack.get_template(template_name)

        # Render the template
        rendered = render_template(template, variables)

        # Get optional components
        system = pack.system_instruction if include_system else None
        examples = pack.get_examples(example_set) if example_set else None

        # Build fingerprint
        fingerprint = PromptFingerprint.from_pack(
            pack=pack,
            template_name=template_name,
            example_set_name=example_set,
        )

        return ResolvedPrompt(
            prompt_text=rendered,
            system_instruction=system,
            examples=examples,
            fingerprint=fingerprint,
        )

    def resolve_rubric(
        self,
        pack_name: str,
        rubric_name: str,
        version: str | None = None,
    ) -> Rubric:
        """
        Resolve a rubric reference.

        Args:
            pack_name: Name of the prompt pack.
            rubric_name: Name of the rubric.
            version: Optional version to validate.

        Returns:
            The resolved Rubric.
        """
        pack = self.load_pack(pack_name, version)
        return pack.get_rubric(rubric_name)

    def resolve_schema(
        self,
        pack_name: str,
        schema_name: str,
        version: str | None = None,
    ) -> OutputSchema:
        """
        Resolve an output schema reference.

        Args:
            pack_name: Name of the prompt pack.
            schema_name: Name of the schema.
            version: Optional version to validate.

        Returns:
            The resolved OutputSchema.
        """
        pack = self.load_pack(pack_name, version)
        return pack.get_output_schema(schema_name)

    def list_packs(self) -> list[str]:
        """
        List available prompt packs.

        Returns:
            List of pack names.
        """
        packs = []
        if self.prompts_dir.exists():
            for path in self.prompts_dir.iterdir():
                if path.is_dir() and (path / "pack.yaml").exists():
                    packs.append(path.name)
        return sorted(packs)

    def validate_pack(self, pack_name: str) -> list[str]:
        """
        Validate a prompt pack.

        Args:
            pack_name: Name of the pack to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        try:
            pack = self.load_pack(pack_name)
        except Exception as e:
            return [f"Failed to load pack: {e}"]

        # Validate templates have required vars
        for name, template in pack.templates.items():
            if not template.template:
                errors.append(f"Template '{name}' has empty template text")

        # Validate rubrics have criteria
        for name, rubric in pack.rubrics.items():
            if not rubric.criteria:
                errors.append(f"Rubric '{name}' has no criteria")

        # Validate output schemas
        for name, schema in pack.output_schemas.items():
            if not schema.schema_def:
                errors.append(f"Output schema '{name}' has empty schema_def")

        return errors


class ResolvedPrompt:
    """
    A fully resolved prompt ready for use.

    Contains concrete strings and fingerprint for caching.
    """

    def __init__(
        self,
        prompt_text: str,
        system_instruction: str | None,
        examples: list[FewShotExample] | None,
        fingerprint: PromptFingerprint,
    ):
        self.prompt_text = prompt_text
        self.system_instruction = system_instruction
        self.examples = examples
        self.fingerprint = fingerprint

    @property
    def hash(self) -> str:
        """Get combined hash for cache key."""
        return self.fingerprint.combined_hash

    def to_messages(
        self,
        user_content: str | list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Convert to OpenAI-style messages format.

        Args:
            user_content: Optional additional user content.
                If string, appended to prompt text.
                If list, used as message content directly.

        Returns:
            List of message dicts with role and content.
        """
        messages: list[dict[str, Any]] = []

        # System message
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})

        # Few-shot examples
        if self.examples:
            for example in self.examples:
                messages.append({"role": "user", "content": example.input})
                messages.append({"role": "assistant", "content": example.output})

        # User message with prompt
        if isinstance(user_content, list):
            # Multimodal content
            messages.append({"role": "user", "content": user_content})
        else:
            # Text content
            content = self.prompt_text
            if user_content:
                content = f"{content}\n\n{user_content}"
            messages.append({"role": "user", "content": content})

        return messages
