"""
VLM Attribute Tagging Stage.

Tags images with structured attributes using Vision-Language Models.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from labelforge.llm.vlm_messages import build_vlm_messages
from labelforge.pipelines.stage import Stage, StageConfig, StageContext, StageType

if TYPE_CHECKING:
    import ray.data


class VLMTagsStage(Stage):
    """
    Stage that tags images with structured attributes using a VLM.

    Supports both open-vocabulary and closed-vocabulary tagging,
    with optional confidence scores.

    Required inputs:
    - image: PIL Image or image bytes
    - row_id: Unique row identifier

    Outputs:
    - tags: List of string tags
    - attributes: Dict of attribute key-value pairs
    - tags_raw: Raw model output before postprocessing
    """

    def __init__(
        self,
        config: StageConfig,
        *,
        system_instruction: str | None = None,
        tag_vocabulary: list[str] | None = None,
        attribute_schema: dict[str, list[str]] | None = None,
        max_tags: int = 10,
        max_tokens: int = 512,
        temperature: float = 0.0,
        use_guided_decoding: bool = True,
    ):
        """
        Initialize VLM tagging stage.

        Args:
            config: Stage configuration.
            system_instruction: Optional system prompt.
            tag_vocabulary: Optional closed vocabulary for tags.
            attribute_schema: Optional dict mapping attribute names to allowed values.
            max_tags: Maximum number of tags to return.
            max_tokens: Maximum tokens in generated response.
            temperature: Sampling temperature (0.0 for deterministic).
            use_guided_decoding: Whether to use JSON schema-guided decoding.
        """
        super().__init__(config)
        self.system_instruction = system_instruction
        self.tag_vocabulary = tag_vocabulary
        self.attribute_schema = attribute_schema or {}
        self.max_tags = max_tags
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_guided_decoding = use_guided_decoding

    @property
    def input_schema(self) -> dict[str, type]:
        """Required input fields."""
        return {
            "image": object,  # PIL Image or bytes
            "row_id": str,
        }

    @property
    def output_schema(self) -> dict[str, type]:
        """Output fields added by this stage."""
        return {
            "tags": list,
            "attributes": dict,
            "tags_raw": str,
        }

    def _build_tagging_prompt(self) -> str:
        """Build the image tagging prompt."""
        prompt_parts = ["Analyze this image and provide:"]

        # Tags section
        if self.tag_vocabulary:
            vocab_list = ", ".join(self.tag_vocabulary)
            prompt_parts.append(
                f"1. Tags: Select relevant tags from this list: [{vocab_list}]"
            )
        else:
            prompt_parts.append(
                f"1. Tags: Up to {self.max_tags} descriptive tags for the image"
            )

        # Attributes section
        if self.attribute_schema:
            prompt_parts.append("2. Attributes:")
            for attr_name, values in self.attribute_schema.items():
                if values:
                    values_list = ", ".join(values)
                    prompt_parts.append(f"   - {attr_name}: one of [{values_list}]")
                else:
                    prompt_parts.append(f"   - {attr_name}: free-form value")
        else:
            prompt_parts.append(
                "2. Attributes: Key descriptive attributes (e.g., style, mood, setting)"
            )

        prompt_parts.append("\nRespond with a JSON object containing 'tags' and 'attributes'.")
        return "\n".join(prompt_parts)

    def _get_output_schema(self) -> dict[str, Any]:
        """Get JSON schema for guided decoding."""
        # Tags schema
        if self.tag_vocabulary:
            tags_schema = {
                "type": "array",
                "items": {"type": "string", "enum": self.tag_vocabulary},
                "maxItems": self.max_tags,
            }
        else:
            tags_schema = {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": self.max_tags,
            }

        # Attributes schema
        if self.attribute_schema:
            attr_properties = {}
            for attr_name, values in self.attribute_schema.items():
                if values:
                    attr_properties[attr_name] = {"type": "string", "enum": values}
                else:
                    attr_properties[attr_name] = {"type": "string"}
            attributes_schema = {
                "type": "object",
                "properties": attr_properties,
            }
        else:
            attributes_schema = {"type": "object"}

        return {
            "type": "object",
            "properties": {
                "tags": tags_schema,
                "attributes": attributes_schema,
            },
            "required": ["tags", "attributes"],
        }

    def _create_preprocess_fn(self, context: StageContext) -> Any:
        """Create preprocessing function for Ray Data LLM."""
        system_instruction = self.system_instruction
        prompt = self._build_tagging_prompt()
        max_tokens = self.max_tokens
        temperature = self.temperature
        use_guided_decoding = self.use_guided_decoding
        output_schema = self._get_output_schema()

        def preprocess(row: dict[str, Any]) -> dict[str, Any]:
            """Prepare row for VLM tagging."""
            image = row.get("image")

            messages = build_vlm_messages(
                text=prompt,
                image=image,
                system_instruction=system_instruction,
            )

            sampling_params: dict[str, Any] = {
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if use_guided_decoding:
                sampling_params["guided_decoding"] = {"json": output_schema}

            return {
                "messages": messages,
                "sampling_params": sampling_params,
                "row_id": row.get("row_id", ""),
            }

        return preprocess

    def _create_postprocess_fn(self) -> Any:
        """Create postprocessing function for Ray Data LLM."""
        max_tags = self.max_tags

        def postprocess(row: dict[str, Any]) -> dict[str, Any]:
            """Process VLM tagging output."""
            generated_text = row.get("generated_text", "")

            result: dict[str, Any] = {
                "row_id": row.get("row_id", ""),
                "tags_raw": generated_text,
                "tags": [],
                "attributes": {},
            }

            try:
                parsed = json.loads(generated_text.strip())
                tags = parsed.get("tags", [])
                # Ensure tags is a list and limit to max
                if isinstance(tags, list):
                    result["tags"] = tags[:max_tags]
                result["attributes"] = parsed.get("attributes", {})
            except (json.JSONDecodeError, TypeError):
                result["parse_error"] = True

            return result

        return postprocess

    def run(
        self,
        dataset: ray.data.Dataset,
        context: StageContext,
    ) -> ray.data.Dataset:
        """Execute the tagging stage on a dataset."""
        from labelforge.core.model_spec import ModelSpec
        from labelforge.llm.processor_factory import apply_processor_to_dataset

        model_spec_name = self.config.model_spec_name
        if model_spec_name:
            model_spec = ModelSpec(model_source=model_spec_name)
        else:
            model_spec = ModelSpec(
                model_source="llava-hf/llava-1.5-7b-hf",
                task_type="generate",
            )

        preprocess_fn = self._create_preprocess_fn(context)
        postprocess_fn = self._create_postprocess_fn()

        return apply_processor_to_dataset(
            dataset=dataset,
            model_spec=model_spec,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
        )


def create_vlm_tags_stage(
    name: str = "vlm_tags",
    *,
    system_instruction: str | None = None,
    tag_vocabulary: list[str] | None = None,
    attribute_schema: dict[str, list[str]] | None = None,
    max_tags: int = 10,
    max_tokens: int = 512,
    temperature: float = 0.0,
    use_guided_decoding: bool = True,
    model_spec_name: str | None = None,
    version: str = "1.0.0",
) -> VLMTagsStage:
    """
    Create a VLM tagging stage with configuration.

    Args:
        name: Stage name.
        system_instruction: Optional system prompt.
        tag_vocabulary: Optional closed vocabulary for tags.
        attribute_schema: Dict mapping attribute names to allowed values.
        max_tags: Maximum number of tags.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
        use_guided_decoding: Use JSON schema-guided decoding.
        model_spec_name: Optional model specification name.
        version: Stage version.

    Returns:
        Configured VLMTagsStage.

    Example:
        >>> stage = create_vlm_tags_stage(
        ...     name="image_tagger",
        ...     tag_vocabulary=["landscape", "portrait", "nature", "urban"],
        ...     attribute_schema={"style": ["photo", "illustration", "painting"]},
        ... )
    """
    config = StageConfig(
        name=name,
        stage_type=StageType.VLM_TAGS,
        version=version,
        model_spec_name=model_spec_name,
        params={
            "tag_vocabulary": tag_vocabulary,
            "attribute_schema": attribute_schema,
            "max_tags": max_tags,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_guided_decoding": use_guided_decoding,
        },
    )

    return VLMTagsStage(
        config=config,
        system_instruction=system_instruction,
        tag_vocabulary=tag_vocabulary,
        attribute_schema=attribute_schema,
        max_tags=max_tags,
        max_tokens=max_tokens,
        temperature=temperature,
        use_guided_decoding=use_guided_decoding,
    )
