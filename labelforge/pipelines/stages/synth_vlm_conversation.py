"""
VLM Conversation Synthesis Stage.

Generates multi-turn conversations grounded in images.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from labelforge.core.synth_spec import SynthRecordType, create_synth_row_id
from labelforge.llm.vlm_messages import build_vlm_messages
from labelforge.pipelines.stage import Stage, StageConfig, StageContext, StageType

if TYPE_CHECKING:
    import ray.data


class VLMConversationStage(Stage):
    """
    Stage that generates multi-turn conversations about images.

    Creates image-grounded conversations for VLM training data.

    Required inputs:
    - image: PIL Image or image bytes
    - row_id: Unique row identifier

    Optional inputs:
    - caption: Existing caption for context

    Outputs:
    - conversation: JSON array of conversation turns
    - synth_row_id: Unique ID for synthetic record
    """

    def __init__(
        self,
        config: StageConfig,
        *,
        num_turns: int = 4,
        caption_field: str | None = "caption",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_instruction: str | None = None,
        conversation_style: str = "educational",
    ):
        """
        Initialize VLM conversation stage.

        Args:
            config: Stage configuration.
            num_turns: Number of conversation turns to generate.
            caption_field: Optional field with existing caption.
            max_tokens: Maximum tokens in generation.
            temperature: Sampling temperature.
            system_instruction: Optional system prompt.
            conversation_style: Style of conversation (educational, casual, etc).
        """
        super().__init__(config)
        self.num_turns = num_turns
        self.caption_field = caption_field
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_instruction = system_instruction
        self.conversation_style = conversation_style

    @property
    def input_schema(self) -> dict[str, type]:
        """Required input fields."""
        return {
            "image": object,
            "row_id": str,
        }

    @property
    def output_schema(self) -> dict[str, type]:
        """Output fields added by this stage."""
        return {
            "conversation": list,
            "synth_row_id": str,
            "turn_count": int,
        }

    def _build_conversation_prompt(self, caption: str | None) -> str:
        """Build the conversation generation prompt."""
        prompt = f"""Generate a {self.num_turns}-turn conversation about this image.

The conversation should be {self.conversation_style} in style.
"""
        if caption:
            prompt += f"\nImage description: {caption}\n"

        prompt += f"""
Generate a JSON object with a "turns" array where each turn has:
- "role": Either "user" or "assistant"
- "content": The message content

The conversation should:
1. Start with a user question about the image
2. Alternate between user and assistant
3. Have {self.num_turns} total turns
4. Be natural and informative

Respond with only the JSON object."""
        return prompt

    def _get_output_schema(self) -> dict[str, Any]:
        """Get JSON schema for guided decoding."""
        return {
            "type": "object",
            "properties": {
                "turns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "enum": ["user", "assistant"],
                            },
                            "content": {"type": "string"},
                        },
                        "required": ["role", "content"],
                    },
                },
            },
            "required": ["turns"],
        }

    def _create_preprocess_fn(self, context: StageContext) -> Any:
        """Create preprocessing function for Ray Data LLM."""
        caption_field = self.caption_field
        max_tokens = self.max_tokens
        temperature = self.temperature
        system_instruction = self.system_instruction
        output_schema = self._get_output_schema()
        build_prompt = self._build_conversation_prompt

        def preprocess(row: dict[str, Any]) -> dict[str, Any]:
            """Prepare row for VLM conversation generation."""
            image = row.get("image")
            caption = row.get(caption_field) if caption_field else None

            prompt = build_prompt(caption)

            # Build VLM messages with image
            messages = build_vlm_messages(
                text=prompt,
                image=image,
                system_instruction=system_instruction,
            )

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

        def postprocess(row: dict[str, Any]) -> dict[str, Any]:
            """Process VLM conversation output."""
            generated_text = row.get("generated_text", "")
            row_id = row.get("row_id", "")

            result = {
                "row_id": row_id,
                "synth_row_id": create_synth_row_id(
                    row_id, SynthRecordType.CONVERSATION, 0
                ),
                "conversation": [],
                "turn_count": 0,
            }

            try:
                parsed = json.loads(generated_text.strip())
                turns = parsed.get("turns", [])
                result["conversation"] = turns
                result["turn_count"] = len(turns)
            except (json.JSONDecodeError, TypeError):
                result["parse_error"] = True
                result["raw_output"] = generated_text

            return result

        return postprocess

    def run(
        self,
        dataset: ray.data.Dataset,
        context: StageContext,
    ) -> ray.data.Dataset:
        """Execute the VLM conversation stage."""
        from labelforge.core.model_spec import ModelSpec
        from labelforge.llm.processor_factory import apply_processor_to_dataset

        model_spec_name = self.config.model_spec_name
        if model_spec_name:
            model_spec = ModelSpec(
                model_source=model_spec_name,
                task_type="generate",
                has_image=True,
            )
        else:
            model_spec = ModelSpec(
                model_source="llava-hf/llava-1.5-7b-hf",
                task_type="generate",
                has_image=True,
            )

        preprocess_fn = self._create_preprocess_fn(context)
        postprocess_fn = self._create_postprocess_fn()

        return apply_processor_to_dataset(
            dataset=dataset,
            model_spec=model_spec,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
        )


def create_vlm_conversation_stage(
    name: str = "synth_vlm_conversation",
    *,
    num_turns: int = 4,
    caption_field: str | None = "caption",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    conversation_style: str = "educational",
    model_spec_name: str | None = None,
    version: str = "1.0.0",
) -> VLMConversationStage:
    """
    Create a VLM conversation synthesis stage.

    Args:
        name: Stage name.
        num_turns: Number of conversation turns.
        caption_field: Field with existing caption.
        max_tokens: Max generation tokens.
        temperature: Sampling temperature.
        conversation_style: Style of conversation.
        model_spec_name: VLM model to use.
        version: Stage version.

    Returns:
        Configured VLMConversationStage.
    """
    config = StageConfig(
        name=name,
        stage_type=StageType.SYNTH,
        version=version,
        model_spec_name=model_spec_name,
        params={
            "num_turns": num_turns,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "conversation_style": conversation_style,
        },
    )

    return VLMConversationStage(
        config=config,
        num_turns=num_turns,
        caption_field=caption_field,
        max_tokens=max_tokens,
        temperature=temperature,
        conversation_style=conversation_style,
    )
