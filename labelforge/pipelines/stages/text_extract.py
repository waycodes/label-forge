"""
Text Extraction Stage.

Extracts structured fields from text using LLMs with schema enforcement.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from labelforge.pipelines.stage import Stage, StageConfig, StageContext, StageType

if TYPE_CHECKING:
    import ray.data


class TextExtractStage(Stage):
    """
    Stage that extracts structured fields from text.

    Uses JSON schema-guided decoding to enforce output structure.
    Supports nested objects, arrays, and optional fields.

    Required inputs:
    - text: Text to extract from (or custom field)
    - row_id: Unique row identifier

    Outputs:
    - extracted: Dict of extracted field values
    - extraction_raw: Raw model output
    """

    def __init__(
        self,
        config: StageConfig,
        *,
        extraction_schema: dict[str, Any],
        text_field: str = "text",
        system_instruction: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        use_guided_decoding: bool = True,
    ):
        """
        Initialize text extraction stage.

        Args:
            config: Stage configuration.
            extraction_schema: JSON Schema defining fields to extract.
            text_field: Name of field containing source text.
            system_instruction: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            use_guided_decoding: Use JSON schema-guided decoding.
        """
        super().__init__(config)
        self.extraction_schema = extraction_schema
        self.text_field = text_field
        self.system_instruction = system_instruction
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_guided_decoding = use_guided_decoding

    @property
    def input_schema(self) -> dict[str, type]:
        """Required input fields."""
        return {
            self.text_field: str,
            "row_id": str,
        }

    @property
    def output_schema(self) -> dict[str, type]:
        """Output fields added by this stage."""
        return {
            "extracted": dict,
            "extraction_raw": str,
        }

    def _build_extraction_prompt(self, text: str) -> str:
        """Build the extraction prompt."""
        # Describe the schema in a readable way
        schema_desc = self._describe_schema(self.extraction_schema)

        return f"""Extract the following information from the text below:

{schema_desc}

Text to analyze:
\"\"\"
{text}
\"\"\"

Respond with a JSON object containing the extracted fields. Use null for fields that cannot be determined from the text."""

    def _describe_schema(self, schema: dict[str, Any], indent: int = 0) -> str:
        """Convert JSON schema to human-readable description."""
        lines = []
        prefix = "  " * indent

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for field_name, field_schema in properties.items():
            field_type = field_schema.get("type", "any")
            description = field_schema.get("description", "")
            is_required = field_name in required

            req_marker = " (required)" if is_required else " (optional)"

            if field_type == "object" and "properties" in field_schema:
                lines.append(f"{prefix}- {field_name}{req_marker}: object with:")
                lines.append(self._describe_schema(field_schema, indent + 1))
            elif field_type == "array":
                items = field_schema.get("items", {})
                item_type = items.get("type", "any")
                lines.append(f"{prefix}- {field_name}{req_marker}: array of {item_type}")
                if description:
                    lines.append(f"{prefix}  Description: {description}")
            else:
                enum_values = field_schema.get("enum")
                if enum_values:
                    values_str = ", ".join(f'"{v}"' for v in enum_values)
                    lines.append(f"{prefix}- {field_name}{req_marker}: one of [{values_str}]")
                else:
                    lines.append(f"{prefix}- {field_name}{req_marker}: {field_type}")
                if description:
                    lines.append(f"{prefix}  Description: {description}")

        return "\n".join(lines)

    def _create_preprocess_fn(self, context: StageContext) -> Any:
        """Create preprocessing function for Ray Data LLM."""
        text_field = self.text_field
        system_instruction = self.system_instruction
        max_tokens = self.max_tokens
        temperature = self.temperature
        use_guided_decoding = self.use_guided_decoding
        extraction_schema = self.extraction_schema
        build_prompt = self._build_extraction_prompt

        def preprocess(row: dict[str, Any]) -> dict[str, Any]:
            text = row.get(text_field, "")
            prompt = build_prompt(text)

            messages: list[dict[str, Any]] = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            sampling_params: dict[str, Any] = {
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if use_guided_decoding:
                sampling_params["guided_decoding"] = {"json": extraction_schema}

            return {
                "messages": messages,
                "sampling_params": sampling_params,
                "row_id": row.get("row_id", ""),
            }

        return preprocess

    def _create_postprocess_fn(self) -> Any:
        """Create postprocessing function for Ray Data LLM."""

        def postprocess(row: dict[str, Any]) -> dict[str, Any]:
            generated_text = row.get("generated_text", "")

            result: dict[str, Any] = {
                "row_id": row.get("row_id", ""),
                "extraction_raw": generated_text,
                "extracted": {},
            }

            try:
                parsed = json.loads(generated_text.strip())
                if isinstance(parsed, dict):
                    result["extracted"] = parsed
            except (json.JSONDecodeError, TypeError):
                result["parse_error"] = True

            return result

        return postprocess

    def run(
        self,
        dataset: ray.data.Dataset,
        context: StageContext,
    ) -> ray.data.Dataset:
        """Execute the extraction stage on a dataset."""
        from labelforge.core.model_spec import ModelSpec
        from labelforge.llm.processor_factory import apply_processor_to_dataset

        model_spec_name = self.config.model_spec_name
        if model_spec_name:
            model_spec = ModelSpec(model_source=model_spec_name)
        else:
            model_spec = ModelSpec(
                model_source="meta-llama/Llama-3.1-8B-Instruct",
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


def create_text_extract_stage(
    name: str = "text_extract",
    *,
    extraction_schema: dict[str, Any],
    text_field: str = "text",
    system_instruction: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    model_spec_name: str | None = None,
    version: str = "1.0.0",
) -> TextExtractStage:
    """
    Create a text extraction stage.

    Args:
        name: Stage name.
        extraction_schema: JSON Schema for fields to extract.
        text_field: Name of field containing source text.
        system_instruction: Optional system prompt.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
        model_spec_name: Optional model spec name.
        version: Stage version.

    Returns:
        Configured TextExtractStage.

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string", "description": "Person's name"},
        ...         "email": {"type": "string", "description": "Email address"},
        ...         "phone": {"type": "string", "description": "Phone number"},
        ...     },
        ...     "required": ["name"],
        ... }
        >>> stage = create_text_extract_stage(
        ...     name="contact_extractor",
        ...     extraction_schema=schema,
        ... )
    """
    config = StageConfig(
        name=name,
        stage_type=StageType.TEXT_EXTRACT,
        version=version,
        model_spec_name=model_spec_name,
        params={
            "extraction_schema": extraction_schema,
            "text_field": text_field,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    return TextExtractStage(
        config=config,
        extraction_schema=extraction_schema,
        text_field=text_field,
        system_instruction=system_instruction,
        max_tokens=max_tokens,
        temperature=temperature,
    )
