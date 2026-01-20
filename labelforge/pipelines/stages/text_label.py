"""
Text Classification Stage.

Classifies text into categories using LLMs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from labelforge.pipelines.stage import Stage, StageConfig, StageContext, StageType

if TYPE_CHECKING:
    import ray.data


class TextLabelStage(Stage):
    """
    Stage that classifies text into predefined categories.

    Supports single-label and multi-label classification with
    optional chain-of-thought reasoning.

    Required inputs:
    - text: Text content to classify (or custom field)
    - row_id: Unique row identifier

    Outputs:
    - labels: List of assigned labels
    - label_primary: Primary label (for single-label mode)
    - reasoning: Optional reasoning explanation
    - label_raw: Raw model output
    """

    def __init__(
        self,
        config: StageConfig,
        *,
        categories: list[str],
        text_field: str = "text",
        multi_label: bool = False,
        max_labels: int = 3,
        include_reasoning: bool = False,
        system_instruction: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        use_guided_decoding: bool = True,
    ):
        """
        Initialize text classification stage.

        Args:
            config: Stage configuration.
            categories: List of valid category labels.
            text_field: Name of field containing text to classify.
            multi_label: If True, allow multiple labels per sample.
            max_labels: Maximum labels in multi-label mode.
            include_reasoning: Include chain-of-thought reasoning.
            system_instruction: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            use_guided_decoding: Use JSON schema-guided decoding.
        """
        super().__init__(config)
        self.categories = categories
        self.text_field = text_field
        self.multi_label = multi_label
        self.max_labels = max_labels
        self.include_reasoning = include_reasoning
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
        schema = {
            "labels": list,
            "label_primary": str,
            "label_raw": str,
        }
        if self.include_reasoning:
            schema["reasoning"] = str
        return schema

    def _build_classification_prompt(self, text: str) -> str:
        """Build the classification prompt."""
        categories_list = ", ".join(f'"{c}"' for c in self.categories)

        if self.multi_label:
            mode_desc = f"one or more categories (up to {self.max_labels})"
        else:
            mode_desc = "exactly one category"

        prompt = f"""Classify the following text into {mode_desc} from this list:
[{categories_list}]

Text to classify:
\"\"\"
{text}
\"\"\"
"""
        if self.include_reasoning:
            prompt += "\nProvide brief reasoning for your classification."

        prompt += """

Respond with a JSON object containing:
- "labels": array of selected category labels
- "primary": the most relevant single label"""

        if self.include_reasoning:
            prompt += '\n- "reasoning": brief explanation'

        return prompt

    def _get_output_schema(self) -> dict[str, Any]:
        """Get JSON schema for guided decoding."""
        labels_schema = {
            "type": "array",
            "items": {"type": "string", "enum": self.categories},
        }
        if self.multi_label:
            labels_schema["minItems"] = 1
            labels_schema["maxItems"] = self.max_labels
        else:
            labels_schema["minItems"] = 1
            labels_schema["maxItems"] = 1

        properties = {
            "labels": labels_schema,
            "primary": {"type": "string", "enum": self.categories},
        }
        required = ["labels", "primary"]

        if self.include_reasoning:
            properties["reasoning"] = {"type": "string"}
            required.append("reasoning")

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _create_preprocess_fn(self, context: StageContext) -> Any:
        """Create preprocessing function for Ray Data LLM."""
        text_field = self.text_field
        system_instruction = self.system_instruction
        max_tokens = self.max_tokens
        temperature = self.temperature
        use_guided_decoding = self.use_guided_decoding
        output_schema = self._get_output_schema()
        build_prompt = self._build_classification_prompt

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
                sampling_params["guided_decoding"] = {"json": output_schema}

            return {
                "messages": messages,
                "sampling_params": sampling_params,
                "row_id": row.get("row_id", ""),
            }

        return preprocess

    def _create_postprocess_fn(self) -> Any:
        """Create postprocessing function for Ray Data LLM."""
        include_reasoning = self.include_reasoning

        def postprocess(row: dict[str, Any]) -> dict[str, Any]:
            generated_text = row.get("generated_text", "")

            result: dict[str, Any] = {
                "row_id": row.get("row_id", ""),
                "label_raw": generated_text,
                "labels": [],
                "label_primary": "",
            }

            if include_reasoning:
                result["reasoning"] = ""

            try:
                parsed = json.loads(generated_text.strip())
                labels = parsed.get("labels", [])
                if isinstance(labels, list):
                    result["labels"] = labels
                result["label_primary"] = parsed.get("primary", labels[0] if labels else "")
                if include_reasoning:
                    result["reasoning"] = parsed.get("reasoning", "")
            except (json.JSONDecodeError, TypeError, IndexError):
                result["parse_error"] = True

            return result

        return postprocess

    def run(
        self,
        dataset: ray.data.Dataset,
        context: StageContext,
    ) -> ray.data.Dataset:
        """Execute the classification stage on a dataset."""
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


def create_text_label_stage(
    name: str = "text_label",
    *,
    categories: list[str],
    text_field: str = "text",
    multi_label: bool = False,
    max_labels: int = 3,
    include_reasoning: bool = False,
    system_instruction: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    model_spec_name: str | None = None,
    version: str = "1.0.0",
) -> TextLabelStage:
    """
    Create a text classification stage.

    Args:
        name: Stage name.
        categories: List of valid category labels.
        text_field: Name of field containing text.
        multi_label: Allow multiple labels.
        max_labels: Maximum labels in multi-label mode.
        include_reasoning: Include CoT reasoning.
        system_instruction: Optional system prompt.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
        model_spec_name: Optional model spec name.
        version: Stage version.

    Returns:
        Configured TextLabelStage.

    Example:
        >>> stage = create_text_label_stage(
        ...     name="sentiment",
        ...     categories=["positive", "negative", "neutral"],
        ...     include_reasoning=True,
        ... )
    """
    config = StageConfig(
        name=name,
        stage_type=StageType.TEXT_LABEL,
        version=version,
        model_spec_name=model_spec_name,
        params={
            "categories": categories,
            "text_field": text_field,
            "multi_label": multi_label,
            "max_labels": max_labels,
            "include_reasoning": include_reasoning,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    return TextLabelStage(
        config=config,
        categories=categories,
        text_field=text_field,
        multi_label=multi_label,
        max_labels=max_labels,
        include_reasoning=include_reasoning,
        system_instruction=system_instruction,
        max_tokens=max_tokens,
        temperature=temperature,
    )
