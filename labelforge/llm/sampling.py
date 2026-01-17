"""
Per-row sampling parameters injection.

Enables ablations without code changes by setting sampling params per row.
"""

from __future__ import annotations

from typing import Any

from labelforge.core.model_spec import SamplingParams
from labelforge.core.seeds import derive_sampling_seed


def inject_sampling_params(
    row: dict[str, Any],
    base_params: SamplingParams,
    row_seed: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Inject sampling parameters into a row for Ray Data LLM.

    Args:
        row: Input row dict.
        base_params: Base sampling parameters.
        row_seed: Optional row-specific seed (overrides base seed).
        overrides: Optional parameter overrides for this row.

    Returns:
        Row dict with sampling_params added.

    Example:
        >>> from labelforge.core.model_spec import SamplingParams
        >>> params = SamplingParams(temperature=0.7, max_tokens=100)
        >>> row = {"text": "Hello", "row_id": "123"}
        >>> result = inject_sampling_params(row, params, row_seed=42)
        >>> result["sampling_params"]["seed"]
        42
    """
    # Start with base params
    params_dict = base_params.to_vllm_params()

    # Apply row-specific seed
    if row_seed is not None:
        params_dict["seed"] = row_seed

    # Apply overrides
    if overrides:
        params_dict.update(overrides)

    # Create output row with sampling params
    result = dict(row)
    result["sampling_params"] = params_dict

    return result


def create_sampling_injector(
    base_params: SamplingParams,
    stage_seed: int | None = None,
    per_row_overrides: dict[str, dict[str, Any]] | None = None,
) -> callable:
    """
    Create a function that injects sampling params into rows.

    Args:
        base_params: Base sampling parameters.
        stage_seed: Optional stage seed for deriving row seeds.
        per_row_overrides: Optional dict mapping row_id to param overrides.

    Returns:
        Function that takes a row and returns it with sampling_params.

    Example:
        >>> from labelforge.core.model_spec import SamplingParams
        >>> params = SamplingParams(temperature=0.7)
        >>> inject = create_sampling_injector(params, stage_seed=42)
        >>> row = {"text": "Hello", "row_id": "123"}
        >>> result = inject(row)
        >>> "sampling_params" in result
        True
    """
    if per_row_overrides is None:
        per_row_overrides = {}

    def inject(row: dict[str, Any]) -> dict[str, Any]:
        row_id = row.get("row_id")

        # Derive row seed if possible
        row_seed = None
        if stage_seed is not None and row_id:
            row_seed = derive_sampling_seed(stage_seed)

        # Get row-specific overrides
        overrides = per_row_overrides.get(row_id) if row_id else None

        return inject_sampling_params(row, base_params, row_seed, overrides)

    return inject


def merge_sampling_params(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge sampling parameter dicts.

    Args:
        base: Base parameters.
        overrides: Override parameters.

    Returns:
        Merged parameters dict.
    """
    result = dict(base)
    result.update(overrides)
    return result


def extract_sampling_params(row: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extract sampling params from a row if present.

    Args:
        row: Input row dict.

    Returns:
        Sampling params dict or None.
    """
    return row.get("sampling_params")


def get_sampling_params_hash(params: dict[str, Any]) -> str:
    """
    Compute hash of sampling params dict.

    Args:
        params: Sampling parameters dict.

    Returns:
        Hex-encoded hash.
    """
    import xxhash
    from labelforge.core.json_canonical import canonical_json_bytes

    json_bytes = canonical_json_bytes(params)
    return xxhash.xxh64(json_bytes).hexdigest()
