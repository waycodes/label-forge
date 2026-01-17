"""
VLM message builder for multimodal content.

Creates OpenAI-style messages with text + image content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


def build_vlm_messages(
    text: str,
    image: Image.Image | None = None,
    system_instruction: str | None = None,
    image_detail: str = "auto",
) -> list[dict[str, Any]]:
    """
    Build OpenAI-style messages for VLM inference.

    Creates a message list with optional system instruction and
    multimodal user content (text + image).

    Args:
        text: User text prompt.
        image: Optional PIL Image to include.
        system_instruction: Optional system message.
        image_detail: Image detail level ("auto", "low", "high").

    Returns:
        List of message dicts for Ray Data LLM.

    Example:
        >>> from PIL import Image
        >>> img = Image.new("RGB", (100, 100))
        >>> messages = build_vlm_messages(
        ...     text="Describe this image",
        ...     image=img,
        ...     system_instruction="You are an image captioner."
        ... )
        >>> messages[0]["role"]
        'system'
        >>> messages[1]["role"]
        'user'
    """
    messages: list[dict[str, Any]] = []

    # System message
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    # User message
    if image is not None:
        # Multimodal content
        content: list[dict[str, Any]] = [
            {"type": "text", "text": text},
            {"type": "image", "image": image},
        ]
        messages.append({"role": "user", "content": content})
    else:
        # Text-only content
        messages.append({"role": "user", "content": text})

    return messages


def build_vlm_messages_with_examples(
    text: str,
    image: Image.Image | None = None,
    system_instruction: str | None = None,
    examples: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build VLM messages with few-shot examples.

    Args:
        text: User text prompt.
        image: Optional PIL Image.
        system_instruction: Optional system message.
        examples: List of {"input": str, "output": str} dicts.

    Returns:
        List of message dicts with examples.
    """
    messages: list[dict[str, Any]] = []

    # System message
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    # Few-shot examples
    if examples:
        for example in examples:
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})

    # User message
    if image is not None:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": text},
            {"type": "image", "image": image},
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": text})

    return messages


def build_vlm_content(
    text: str,
    images: list[Image.Image] | None = None,
) -> list[dict[str, Any]]:
    """
    Build multimodal content list for a user message.

    Args:
        text: Text content.
        images: Optional list of PIL Images.

    Returns:
        Content list for user message.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]

    if images:
        for image in images:
            content.append({"type": "image", "image": image})

    return content


def extract_text_from_messages(messages: list[dict[str, Any]]) -> str:
    """
    Extract text content from messages.

    Useful for logging/debugging without images.

    Args:
        messages: List of message dicts.

    Returns:
        Concatenated text content.
    """
    texts = []

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(f"[{msg['role']}] {content}")
        elif isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            texts.append(f"[{msg['role']}] {' '.join(text_parts)}")

    return "\n".join(texts)
