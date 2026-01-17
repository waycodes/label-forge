"""
LLM processor factory for Ray Data LLM integration.

Creates build_processor configs from model spec + prompt spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import ray.data

from labelforge.core.model_spec import ModelSpec, SamplingParams


def create_processor_config(
    model_spec: ModelSpec,
    preprocess_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    postprocess_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Create a vLLMEngineProcessorConfig-compatible configuration dict.

    This can be passed to ray.data.llm.build_processor() along with
    preprocess and postprocess functions.

    Args:
        model_spec: Model specification.
        preprocess_fn: Optional preprocessing function.
        postprocess_fn: Optional postprocessing function.

    Returns:
        Configuration dict for Ray Data LLM.

    Example:
        >>> from labelforge.core.model_spec import ModelSpec
        >>> spec = ModelSpec(model_source="meta-llama/Llama-2-7b-hf")
        >>> config = create_processor_config(spec)
        >>> config["model"]
        'meta-llama/Llama-2-7b-hf'
    """
    return model_spec.to_vllm_config()


def build_llm_processor(
    model_spec: ModelSpec,
    preprocess_fn: Callable[[dict[str, Any]], dict[str, Any]],
    postprocess_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> Any:
    """
    Build a Ray Data LLM processor from model spec.

    Uses ray.data.llm.build_processor with vLLMEngineProcessorConfig.

    Args:
        model_spec: Model specification.
        preprocess_fn: Preprocessing function that prepares input rows.
        postprocess_fn: Postprocessing function that handles outputs.

    Returns:
        Ray Data LLM Processor instance.

    Raises:
        ImportError: If Ray Data LLM is not available.

    Example:
        >>> def preprocess(row):
        ...     return {"messages": [{"role": "user", "content": row["text"]}]}
        >>> def postprocess(row):
        ...     return {"output": row["generated_text"]}
        >>> processor = build_llm_processor(model_spec, preprocess, postprocess)
    """
    try:
        from ray.data.llm import build_processor, vLLMEngineProcessorConfig
    except ImportError as e:
        raise ImportError(
            "Ray Data LLM not available. Install with: pip install 'ray[data]'"
        ) from e

    # Create the engine config
    config_dict = model_spec.to_vllm_config()

    # Build vLLMEngineProcessorConfig
    engine_config = vLLMEngineProcessorConfig(**config_dict)

    # Build and return processor
    return build_processor(
        engine_config,
        preprocess=preprocess_fn,
        postprocess=postprocess_fn,
    )


def create_preprocess_fn(
    system_instruction: str | None = None,
    prompt_template: str | None = None,
    default_sampling_params: SamplingParams | None = None,
    text_field: str = "text",
    include_images: bool = False,
    image_field: str = "image",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Create a preprocessing function for Ray Data LLM.

    Args:
        system_instruction: Optional system message.
        prompt_template: Optional template with {text} placeholder.
        default_sampling_params: Default sampling parameters.
        text_field: Name of text field in input row.
        include_images: Whether to include images in messages.
        image_field: Name of image field in input row.

    Returns:
        Preprocessing function suitable for build_processor.

    Example:
        >>> preprocess = create_preprocess_fn(
        ...     system_instruction="You are a helpful assistant.",
        ...     prompt_template="Describe this: {text}",
        ... )
        >>> result = preprocess({"text": "a cat", "row_id": "123"})
        >>> "messages" in result
        True
    """
    default_params = (
        default_sampling_params.to_vllm_params()
        if default_sampling_params
        else {}
    )

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        # Build messages
        messages: list[dict[str, Any]] = []

        # System message
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        # User message content
        text = row.get(text_field, "")
        if prompt_template:
            text = prompt_template.format(text=text)

        if include_images and image_field in row:
            # Multimodal content
            content: list[dict[str, Any]] = [{"type": "text", "text": text}]
            image = row.get(image_field)
            if image is not None:
                content.append({"type": "image", "image": image})
            messages.append({"role": "user", "content": content})
        else:
            # Text-only content
            messages.append({"role": "user", "content": text})

        result: dict[str, Any] = {"messages": messages}

        # Add sampling params (can be overridden per-row)
        row_params = row.get("sampling_params", {})
        merged_params = {**default_params, **row_params}
        if merged_params:
            result["sampling_params"] = merged_params

        # Preserve row_id for tracking
        if "row_id" in row:
            result["row_id"] = row["row_id"]

        return result

    return preprocess


def create_postprocess_fn(
    output_field: str = "output",
    parse_json: bool = False,
    preserve_fields: list[str] | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Create a postprocessing function for Ray Data LLM.

    Args:
        output_field: Name for the output field.
        parse_json: Whether to parse output as JSON.
        preserve_fields: Fields to pass through from input.

    Returns:
        Postprocessing function suitable for build_processor.

    Example:
        >>> postprocess = create_postprocess_fn(output_field="caption")
        >>> result = postprocess({"generated_text": "A fluffy cat", "row_id": "123"})
        >>> result["caption"]
        'A fluffy cat'
    """
    if preserve_fields is None:
        preserve_fields = ["row_id"]

    def postprocess(row: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}

        # Preserve specified fields
        for field in preserve_fields:
            if field in row:
                result[field] = row[field]

        # Extract generated text
        generated_text = row.get("generated_text", "")

        if parse_json and generated_text:
            try:
                import json
                result[output_field] = json.loads(generated_text)
            except json.JSONDecodeError:
                result[output_field] = generated_text
                result["parse_error"] = True
        else:
            result[output_field] = generated_text

        return result

    return postprocess


def apply_processor_to_dataset(
    dataset: ray.data.Dataset,
    model_spec: ModelSpec,
    preprocess_fn: Callable[[dict[str, Any]], dict[str, Any]],
    postprocess_fn: Callable[[dict[str, Any]], dict[str, Any]],
    **map_batches_kwargs: Any,
) -> ray.data.Dataset:
    """
    Apply an LLM processor to a Ray Dataset.

    This is a convenience wrapper that creates the processor and
    applies it to the dataset.

    Args:
        dataset: Input Ray Dataset.
        model_spec: Model specification.
        preprocess_fn: Preprocessing function.
        postprocess_fn: Postprocessing function.
        **map_batches_kwargs: Additional kwargs for map_batches.

    Returns:
        Processed Ray Dataset.
    """
    processor = build_llm_processor(model_spec, preprocess_fn, postprocess_fn)

    # Apply processor to dataset
    return dataset.map_batches(
        processor,
        concurrency=model_spec.concurrency,
        **map_batches_kwargs,
    )
