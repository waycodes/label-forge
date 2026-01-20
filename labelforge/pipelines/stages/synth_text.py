"""
Text Synthesis Stage.

Generates synthetic text data like QA pairs and instructions from labels.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from labelforge.core.synth_spec import SynthRecordType, create_synth_row_id
from labelforge.pipelines.stage import Stage, StageConfig, StageContext, StageType

if TYPE_CHECKING:
    import ray.data


class TextSynthStage(Stage):
    """
    Stage that generates synthetic text data from existing labels.

    Supports:
    - QA pair generation from text/labels
    - Instruction-response generation
    - Paraphrase generation

    Required inputs:
    - text: Source text content
    - row_id: Unique row identifier

    Optional inputs:
    - label: Class label for conditioning
    - caption: Caption for context

    Outputs:
    - synth_content: Generated synthetic content (JSON)
    - synth_type: Type of synthetic record
    - synth_row_id: Unique ID for synthetic record
    """

    def __init__(
        self,
        config: StageConfig,
        *,
        synth_type: SynthRecordType = SynthRecordType.QA_PAIR,
        source_field: str = "text",
        context_field: str | None = None,
        label_field: str | None = None,
        num_variants: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_instruction: str | None = None,
    ):
        """
        Initialize text synthesis stage.

        Args:
            config: Stage configuration.
            synth_type: Type of synthetic data to generate.
            source_field: Field containing source text.
            context_field: Optional field for additional context.
            label_field: Optional field for class label conditioning.
            num_variants: Number of variants to generate per input.
            max_tokens: Maximum tokens in generation.
            temperature: Sampling temperature.
            system_instruction: Optional system prompt.
        """
        super().__init__(config)
        self.synth_type = synth_type
        self.source_field = source_field
        self.context_field = context_field
        self.label_field = label_field
        self.num_variants = num_variants
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_instruction = system_instruction

    @property
    def input_schema(self) -> dict[str, type]:
        """Required input fields."""
        return {
            self.source_field: str,
            "row_id": str,
        }

    @property
    def output_schema(self) -> dict[str, type]:
        """Output fields added by this stage."""
        return {
            "synth_content": dict,
            "synth_type": str,
            "synth_row_id": str,
        }

    def _build_prompt(self, row: dict[str, Any]) -> str:
        """Build the synthesis prompt based on type."""
        source_text = row.get(self.source_field, "")
        context = row.get(self.context_field, "") if self.context_field else ""
        label = row.get(self.label_field, "") if self.label_field else ""

        if self.synth_type == SynthRecordType.QA_PAIR:
            return self._build_qa_prompt(source_text, context, label)
        elif self.synth_type == SynthRecordType.INSTRUCTION:
            return self._build_instruction_prompt(source_text, context, label)
        elif self.synth_type == SynthRecordType.PARAPHRASE:
            return self._build_paraphrase_prompt(source_text)
        else:
            return self._build_qa_prompt(source_text, context, label)

    def _build_qa_prompt(
        self, source: str, context: str, label: str
    ) -> str:
        """Build QA generation prompt."""
        prompt = f"""Generate a question-answer pair based on the following content.

Content: {source}
"""
        if context:
            prompt += f"\nContext: {context}"
        if label:
            prompt += f"\nCategory: {label}"

        prompt += """

Generate a JSON object with:
- "question": A natural question about the content
- "answer": The correct answer based on the content

Respond with only the JSON object."""
        return prompt

    def _build_instruction_prompt(
        self, source: str, context: str, label: str
    ) -> str:
        """Build instruction generation prompt."""
        prompt = f"""Generate an instruction-response pair based on the following content.

Content: {source}
"""
        if context:
            prompt += f"\nContext: {context}"
        if label:
            prompt += f"\nCategory: {label}"

        prompt += """

Generate a JSON object with:
- "instruction": A natural user instruction or request
- "response": An appropriate assistant response

Respond with only the JSON object."""
        return prompt

    def _build_paraphrase_prompt(self, source: str) -> str:
        """Build paraphrase prompt."""
        return f"""Paraphrase the following text while preserving its meaning.

Original: {source}

Generate a JSON object with:
- "original": The original text
- "paraphrase": The paraphrased version

Respond with only the JSON object."""

    def _get_output_schema(self) -> dict[str, Any]:
        """Get JSON schema for guided decoding."""
        if self.synth_type == SynthRecordType.QA_PAIR:
            return {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
            }
        elif self.synth_type == SynthRecordType.INSTRUCTION:
            return {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "response": {"type": "string"},
                },
                "required": ["instruction", "response"],
            }
        else:
            return {
                "type": "object",
                "properties": {
                    "original": {"type": "string"},
                    "paraphrase": {"type": "string"},
                },
                "required": ["original", "paraphrase"],
            }

    def _create_preprocess_fn(self, context: StageContext) -> Any:
        """Create preprocessing function for Ray Data LLM."""
        synth_type = self.synth_type
        source_field = self.source_field
        context_field = self.context_field
        label_field = self.label_field
        max_tokens = self.max_tokens
        temperature = self.temperature
        system_instruction = self.system_instruction
        output_schema = self._get_output_schema()

        build_qa = self._build_qa_prompt
        build_inst = self._build_instruction_prompt
        build_para = self._build_paraphrase_prompt

        def preprocess(row: dict[str, Any]) -> dict[str, Any]:
            """Prepare row for synthesis."""
            source_text = row.get(source_field, "")
            context_text = row.get(context_field, "") if context_field else ""
            label = row.get(label_field, "") if label_field else ""

            if synth_type == SynthRecordType.QA_PAIR:
                prompt = build_qa(source_text, context_text, label)
            elif synth_type == SynthRecordType.INSTRUCTION:
                prompt = build_inst(source_text, context_text, label)
            else:
                prompt = build_para(source_text)

            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            return {
                "messages": messages,
                "sampling_params": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "guided_decoding": {"json": output_schema},
                },
                "row_id": row.get("row_id", ""),
            }

        return preprocess

    def _create_postprocess_fn(self) -> Any:
        """Create postprocessing function for Ray Data LLM."""
        synth_type = self.synth_type

        def postprocess(row: dict[str, Any]) -> dict[str, Any]:
            """Process synthesis output."""
            generated_text = row.get("generated_text", "")
            row_id = row.get("row_id", "")

            result = {
                "row_id": row_id,
                "synth_type": synth_type.value,
                "synth_row_id": create_synth_row_id(row_id, synth_type, 0),
                "synth_content": {},
            }

            try:
                content = json.loads(generated_text.strip())
                result["synth_content"] = content
            except (json.JSONDecodeError, TypeError):
                result["synth_content"] = {"raw": generated_text}
                result["parse_error"] = True

            return result

        return postprocess

    def run(
        self,
        dataset: ray.data.Dataset,
        context: StageContext,
    ) -> ray.data.Dataset:
        """Execute the synthesis stage on a dataset."""
        from labelforge.core.model_spec import ModelSpec
        from labelforge.llm.processor_factory import apply_processor_to_dataset

        model_spec_name = self.config.model_spec_name
        if model_spec_name:
            model_spec = ModelSpec(model_source=model_spec_name)
        else:
            model_spec = ModelSpec(
                model_source="meta-llama/Llama-2-7b-chat-hf",
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


def create_text_synth_stage(
    name: str = "synth_text",
    *,
    synth_type: SynthRecordType = SynthRecordType.QA_PAIR,
    source_field: str = "text",
    context_field: str | None = None,
    label_field: str | None = None,
    num_variants: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.7,
    model_spec_name: str | None = None,
    version: str = "1.0.0",
) -> TextSynthStage:
    """
    Create a text synthesis stage.

    Args:
        name: Stage name.
        synth_type: Type of synthetic data.
        source_field: Field containing source text.
        context_field: Optional context field.
        label_field: Optional label field.
        num_variants: Variants per input.
        max_tokens: Max generation tokens.
        temperature: Sampling temperature.
        model_spec_name: Model to use.
        version: Stage version.

    Returns:
        Configured TextSynthStage.
    """
    config = StageConfig(
        name=name,
        stage_type=StageType.SYNTH,
        version=version,
        model_spec_name=model_spec_name,
        params={
            "synth_type": synth_type.value,
            "source_field": source_field,
            "num_variants": num_variants,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    return TextSynthStage(
        config=config,
        synth_type=synth_type,
        source_field=source_field,
        context_field=context_field,
        label_field=label_field,
        num_variants=num_variants,
        max_tokens=max_tokens,
        temperature=temperature,
    )
