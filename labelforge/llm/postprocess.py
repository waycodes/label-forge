"""
Robust postprocessing and parsing for model outputs.

Converts model outputs into validated structured records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


@dataclass
class ParseResult:
    """Result of parsing model output."""

    success: bool
    data: Any
    error: str | None = None
    raw_text: str = ""


def parse_json_output(
    text: str,
    schema: type[BaseModel] | None = None,
    strict: bool = False,
) -> ParseResult:
    """
    Parse JSON from model output text.

    Args:
        text: Raw model output text.
        schema: Optional Pydantic model for validation.
        strict: If True, fail on validation errors.

    Returns:
        ParseResult with parsed data or error.

    Example:
        >>> result = parse_json_output('{"name": "test", "value": 42}')
        >>> result.success
        True
        >>> result.data
        {'name': 'test', 'value': 42}
    """
    # Try to extract JSON from text
    text = text.strip()

    # Handle markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return ParseResult(
            success=False,
            data=None,
            error=f"JSON parse error: {e}",
            raw_text=text,
        )

    # Validate against schema if provided
    if schema is not None:
        try:
            validated = schema.model_validate(data)
            data = validated.model_dump()
        except ValidationError as e:
            if strict:
                return ParseResult(
                    success=False,
                    data=data,
                    error=f"Schema validation error: {e}",
                    raw_text=text,
                )
            # Non-strict mode: return parsed data with warning

    return ParseResult(success=True, data=data, raw_text=text)


def extract_generated_text(row: dict[str, Any]) -> str:
    """
    Extract generated text from Ray Data LLM output row.

    Args:
        row: Output row from Ray Data LLM.

    Returns:
        Generated text string.
    """
    return row.get("generated_text", "")


def create_postprocess_with_parsing(
    output_field: str = "parsed_output",
    schema: type[BaseModel] | None = None,
    strict: bool = False,
    preserve_fields: list[str] | None = None,
) -> callable:
    """
    Create a postprocessing function with JSON parsing.

    Args:
        output_field: Name for parsed output field.
        schema: Optional Pydantic model for validation.
        strict: If True, mark as error on parse failure.
        preserve_fields: Fields to preserve from input.

    Returns:
        Postprocessing function.

    Example:
        >>> postprocess = create_postprocess_with_parsing(output_field="result")
        >>> row = {"generated_text": '{"score": 5}', "row_id": "123"}
        >>> result = postprocess(row)
        >>> result["result"]
        {'score': 5}
    """
    if preserve_fields is None:
        preserve_fields = ["row_id"]

    def postprocess(row: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}

        # Preserve fields
        for field in preserve_fields:
            if field in row:
                result[field] = row[field]

        # Extract and parse
        text = extract_generated_text(row)
        parse_result = parse_json_output(text, schema, strict)

        if parse_result.success:
            result[output_field] = parse_result.data
        else:
            result[output_field] = None
            result["parse_error"] = parse_result.error

        result["raw_output"] = parse_result.raw_text

        return result

    return postprocess


def validate_output_schema(
    data: dict[str, Any],
    required_fields: list[str],
) -> tuple[bool, list[str]]:
    """
    Validate that output has required fields.

    Args:
        data: Output data dict.
        required_fields: List of required field names.

    Returns:
        Tuple of (valid, list of missing fields).
    """
    missing = [f for f in required_fields if f not in data]
    return len(missing) == 0, missing


def clean_output_text(text: str) -> str:
    """
    Clean model output text.

    Removes common artifacts like leading/trailing whitespace,
    markdown formatting, etc.

    Args:
        text: Raw model output.

    Returns:
        Cleaned text.
    """
    text = text.strip()

    # Remove common prefixes
    prefixes = ["Answer:", "Output:", "Response:"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return text
