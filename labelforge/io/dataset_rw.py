"""
Dataset Read/Write Utilities.

Provides standardized I/O for reading and writing datasets in various formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow
    import ray.data


# Supported dataset formats
DATASET_FORMATS = {"parquet", "json", "jsonl", "csv"}


def read_dataset(
    path: str | Path,
    *,
    format: str | None = None,
    schema: pyarrow.Schema | None = None,
    parallelism: int = -1,
    **kwargs: Any,
) -> ray.data.Dataset:
    """
    Read a dataset from various formats using Ray Data.

    Automatically detects format from file extension if not specified.

    Args:
        path: Path to dataset file or directory.
        format: Optional format override ("parquet", "json", "jsonl", "csv").
        schema: Optional PyArrow schema for validation.
        parallelism: Number of read tasks (-1 for auto).
        **kwargs: Additional arguments passed to Ray Data reader.

    Returns:
        Ray Dataset.

    Raises:
        ValueError: If format is unsupported.

    Example:
        >>> ds = read_dataset("/data/images.parquet")
        >>> ds = read_dataset("/data/captions.jsonl", format="jsonl")
    """
    import ray.data

    path = Path(path)

    # Auto-detect format from extension
    if format is None:
        suffix = path.suffix.lower().lstrip(".")
        if suffix in {"jsonl", "ndjson"}:
            format = "jsonl"
        elif suffix in DATASET_FORMATS:
            format = suffix
        else:
            # Default to parquet for directories
            format = "parquet" if path.is_dir() else "json"

    # Read based on format
    if format == "parquet":
        ds = ray.data.read_parquet(
            str(path),
            parallelism=parallelism,
            **kwargs,
        )
    elif format in {"json", "jsonl"}:
        ds = ray.data.read_json(
            str(path),
            parallelism=parallelism,
            **kwargs,
        )
    elif format == "csv":
        ds = ray.data.read_csv(
            str(path),
            parallelism=parallelism,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported format: {format}. Supported: {DATASET_FORMATS}")

    # Validate schema if provided
    if schema is not None:
        ds_schema = ds.schema()
        if ds_schema is not None:
            validate_schema_compatibility(ds_schema, schema)

    return ds


def write_dataset(
    dataset: ray.data.Dataset,
    path: str | Path,
    *,
    format: str = "parquet",
    partition_cols: list[str] | None = None,
    compression: str | None = None,
    num_rows_per_file: int | None = None,
    **kwargs: Any,
) -> None:
    """
    Write a dataset to various formats using Ray Data.

    Args:
        dataset: Ray Dataset to write.
        path: Output path (file or directory).
        format: Output format ("parquet", "json", "jsonl", "csv").
        partition_cols: Columns to partition by (parquet only).
        compression: Compression codec.
        num_rows_per_file: Target rows per output file.
        **kwargs: Additional arguments passed to Ray Data writer.

    Raises:
        ValueError: If format is unsupported.

    Example:
        >>> write_dataset(ds, "/output/captions.parquet")
        >>> write_dataset(ds, "/output/captions.jsonl", format="jsonl")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_kwargs: dict[str, Any] = {**kwargs}
    if num_rows_per_file:
        write_kwargs["num_rows_per_file"] = num_rows_per_file

    if format == "parquet":
        parquet_kwargs: dict[str, Any] = {**write_kwargs}
        if partition_cols:
            parquet_kwargs["partition_cols"] = partition_cols
        if compression:
            parquet_kwargs["compression"] = compression
        dataset.write_parquet(str(path), **parquet_kwargs)
    elif format in {"json", "jsonl"}:
        dataset.write_json(str(path), **write_kwargs)
    elif format == "csv":
        dataset.write_csv(str(path), **write_kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}. Supported: {DATASET_FORMATS}")


def validate_schema_compatibility(
    actual: pyarrow.Schema,
    expected: pyarrow.Schema,
    strict: bool = False,
) -> list[str]:
    """
    Validate compatibility between actual and expected schemas.

    Args:
        actual: The actual schema from data.
        expected: The expected/required schema.
        strict: If True, actual must exactly match expected.

    Returns:
        List of validation error messages.

    Example:
        >>> import pyarrow as pa
        >>> actual = pa.schema([("name", pa.string()), ("age", pa.int64())])
        >>> expected = pa.schema([("name", pa.string())])
        >>> errors = validate_schema_compatibility(actual, expected)
        >>> len(errors)
        0
    """
    errors = []
    actual_fields = {f.name: f for f in actual}
    expected_fields = {f.name: f for f in expected}

    # Check for missing required fields
    for name, expected_field in expected_fields.items():
        if name not in actual_fields:
            errors.append(f"Missing required field: {name}")
        elif strict:
            actual_field = actual_fields[name]
            if actual_field.type != expected_field.type:
                errors.append(
                    f"Field '{name}' type mismatch: "
                    f"expected {expected_field.type}, got {actual_field.type}"
                )

    # In strict mode, check for extra fields
    if strict:
        extra = set(actual_fields.keys()) - set(expected_fields.keys())
        if extra:
            errors.append(f"Unexpected fields: {extra}")

    return errors


def get_schema_fingerprint(schema: pyarrow.Schema) -> str:
    """
    Compute a stable fingerprint for a PyArrow schema.

    Args:
        schema: PyArrow schema to fingerprint.

    Returns:
        Hex-encoded hash.

    Example:
        >>> import pyarrow as pa
        >>> schema = pa.schema([("name", pa.string())])
        >>> fp = get_schema_fingerprint(schema)
        >>> len(fp) == 16
        True
    """
    import xxhash

    from labelforge.core.json_canonical import canonical_json_bytes

    # Convert schema to serializable format
    schema_dict = {
        "fields": [
            {"name": f.name, "type": str(f.type), "nullable": f.nullable}
            for f in schema
        ]
    }

    json_bytes = canonical_json_bytes(schema_dict)
    return xxhash.xxh64(json_bytes).hexdigest()


def read_jsonl_manifest(path: str | Path) -> list[dict[str, Any]]:
    """
    Read a JSONL manifest file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of parsed JSON records.

    Example:
        >>> records = read_jsonl_manifest("/path/to/manifest.jsonl")
    """
    path = Path(path)
    records = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def write_jsonl_manifest(
    records: list[dict[str, Any]],
    path: str | Path,
    *,
    append: bool = False,
) -> None:
    """
    Write records to a JSONL manifest file.

    Args:
        records: List of records to write.
        path: Output path.
        append: If True, append to existing file.

    Example:
        >>> records = [{"row_id": "1", "status": "ok"}]
        >>> write_jsonl_manifest(records, "/path/to/manifest.jsonl")
    """
    from labelforge.core.json_canonical import canonical_json_dumps

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with path.open(mode) as f:
        for record in records:
            f.write(canonical_json_dumps(record) + "\n")


def count_dataset_rows(dataset: ray.data.Dataset) -> int:
    """
    Count the number of rows in a dataset.

    Args:
        dataset: Ray Dataset.

    Returns:
        Row count.
    """
    return dataset.count()


def sample_dataset(
    dataset: ray.data.Dataset,
    n: int,
    *,
    seed: int | None = None,
) -> ray.data.Dataset:
    """
    Sample n rows from a dataset.

    Args:
        dataset: Ray Dataset.
        n: Number of rows to sample.
        seed: Random seed for reproducibility.

    Returns:
        Sampled dataset.
    """
    # Compute sampling fraction
    total = dataset.count()
    if total == 0:
        return dataset
    fraction = min(1.0, n / total)
    return dataset.random_sample(fraction, seed=seed)


def limit_dataset(dataset: ray.data.Dataset, n: int) -> ray.data.Dataset:
    """
    Limit dataset to first n rows.

    Args:
        dataset: Ray Dataset.
        n: Maximum rows.

    Returns:
        Limited dataset.
    """
    return dataset.limit(n)


def filter_dataset(
    dataset: ray.data.Dataset,
    predicate: str,
    *,
    field: str | None = None,
    value: Any | None = None,
) -> ray.data.Dataset:
    """
    Filter dataset rows based on a predicate.

    Args:
        dataset: Ray Dataset.
        predicate: Filter predicate ("eq", "ne", "lt", "gt", "contains").
        field: Field to filter on (required for most predicates).
        value: Value to compare against.

    Returns:
        Filtered dataset.

    Example:
        >>> filtered = filter_dataset(ds, "eq", field="status", value="ok")
    """
    if field is None:
        raise ValueError("Field required for filtering")

    if predicate == "eq":

        def filter_fn(row: dict[str, Any]) -> bool:
            return row.get(field) == value

    elif predicate == "ne":

        def filter_fn(row: dict[str, Any]) -> bool:
            return row.get(field) != value

    elif predicate == "lt":

        def filter_fn(row: dict[str, Any]) -> bool:
            return row.get(field, 0) < value

    elif predicate == "gt":

        def filter_fn(row: dict[str, Any]) -> bool:
            return row.get(field, 0) > value

    elif predicate == "contains":

        def filter_fn(row: dict[str, Any]) -> bool:
            field_value = row.get(field, "")
            return value in field_value if isinstance(field_value, str) else False

    else:
        raise ValueError(f"Unknown predicate: {predicate}")

    return dataset.filter(filter_fn)
