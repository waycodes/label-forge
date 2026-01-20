"""
VLM Image Captioning Stage.

Generates captions for images using Vision-Language Models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from labelforge.llm.vlm_messages import build_vlm_messages
from labelforge.pipelines.stage import Stage, StageConfig, StageContext, StageType

if TYPE_CHECKING:
    import ray.data


class VLMCaptionStage(Stage):
    """
    Stage that generates captions for images using a VLM.

    This stage requires:
    - image: PIL Image or image bytes
    - row_id: Unique row identifier

    Outputs:
    - caption: Generated text caption
    - caption_raw: Raw model output before postprocessing
    """

    def __init__(
        self,
        config: StageConfig,
        *,
        system_instruction: str | None = None,
        prompt_template: str = "Describe this image in detail.",
        max_tokens: int = 256,
        temperature: float = 0.0,
        use_few_shot: bool = False,
    ):
        """
        Initialize VLM caption stage.

        Args:
            config: Stage configuration.
            system_instruction: Optional system prompt for the VLM.
            prompt_template: Prompt to use for captioning.
            max_tokens: Maximum tokens in generated caption.
            temperature: Sampling temperature (0.0 for deterministic).
            use_few_shot: Whether to use few-shot examples from prompt pack.
        """
        super().__init__(config)
        self.system_instruction = system_instruction
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_few_shot = use_few_shot

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
            "caption": str,
            "caption_raw": str,
        }

    def _create_preprocess_fn(
        self,
        context: StageContext,
    ) -> Any:
        """Create preprocessing function for Ray Data LLM."""
        system_instruction = self.system_instruction
        prompt_template = self.prompt_template
        max_tokens = self.max_tokens
        temperature = self.temperature

        def preprocess(row: dict[str, Any]) -> dict[str, Any]:
            """Prepare row for VLM inference."""
            image = row.get("image")

            # Build VLM messages
            messages = build_vlm_messages(
                text=prompt_template,
                image=image,
                system_instruction=system_instruction,
            )

            return {
                "messages": messages,
                "sampling_params": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                "row_id": row.get("row_id", ""),
            }

        return preprocess

    def _create_postprocess_fn(self) -> Any:
        """Create postprocessing function for Ray Data LLM."""

        def postprocess(row: dict[str, Any]) -> dict[str, Any]:
            """Process VLM output."""
            generated_text = row.get("generated_text", "")

            # Clean up caption (remove leading/trailing whitespace)
            caption = generated_text.strip()

            return {
                "row_id": row.get("row_id", ""),
                "caption": caption,
                "caption_raw": generated_text,
            }

        return postprocess

    def run(
        self,
        dataset: ray.data.Dataset,
        context: StageContext,
    ) -> ray.data.Dataset:
        """
        Execute the captioning stage on a dataset.

        Args:
            dataset: Input Ray Dataset with images.
            context: Execution context.

        Returns:
            Dataset with added caption fields.
        """
        from labelforge.core.model_spec import ModelSpec
        from labelforge.llm.processor_factory import apply_processor_to_dataset

        # Create model spec from config or context
        model_spec_name = self.config.model_spec_name
        if model_spec_name:
            # Load from model registry (placeholder)
            model_spec = ModelSpec(model_source=model_spec_name)
        else:
            # Use default VLM model
            model_spec = ModelSpec(
                model_source="llava-hf/llava-1.5-7b-hf",
                task_type="generate",
            )

        # Create preprocessing and postprocessing functions
        preprocess_fn = self._create_preprocess_fn(context)
        postprocess_fn = self._create_postprocess_fn()

        # Apply processor to dataset
        return apply_processor_to_dataset(
            dataset=dataset,
            model_spec=model_spec,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
        )


def create_vlm_caption_stage(
    name: str = "vlm_caption",
    *,
    system_instruction: str | None = None,
    prompt_template: str = "Describe this image in detail.",
    max_tokens: int = 256,
    temperature: float = 0.0,
    model_spec_name: str | None = None,
    version: str = "1.0.0",
) -> VLMCaptionStage:
    """
    Create a VLM captioning stage with configuration.

    Args:
        name: Stage name.
        system_instruction: Optional system prompt.
        prompt_template: Prompt to use for captioning.
        max_tokens: Maximum tokens in generated caption.
        temperature: Sampling temperature.
        model_spec_name: Optional model specification name.
        version: Stage version.

    Returns:
        Configured VLMCaptionStage.

    Example:
        >>> stage = create_vlm_caption_stage(
        ...     name="image_captioner",
        ...     system_instruction="You are an image description expert.",
        ...     max_tokens=512,
        ... )
        >>> stage.name
        'image_captioner'
    """
    config = StageConfig(
        name=name,
        stage_type=StageType.VLM_CAPTION,
        version=version,
        model_spec_name=model_spec_name,
        params={
            "prompt_template": prompt_template,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    return VLMCaptionStage(
        config=config,
        system_instruction=system_instruction,
        prompt_template=prompt_template,
        max_tokens=max_tokens,
        temperature=temperature,
    )
