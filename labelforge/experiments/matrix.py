"""
Ablation experiment matrix.

Generates experiment variants from base config with override combinations.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import xxhash

from labelforge.core.json_canonical import canonical_json_bytes, canonical_json_dumps


@dataclass
class AblationVariant:
    """A single experiment variant in an ablation study."""

    variant_id: str
    variant_name: str
    base_config_hash: str
    overrides: dict[str, Any]
    override_hash: str

    @property
    def unique_id(self) -> str:
        """Unique identifier combining base and overrides."""
        return f"{self.base_config_hash[:8]}_{self.override_hash[:8]}"

    def apply_to_config(self, base_config: dict[str, Any]) -> dict[str, Any]:
        """
        Apply overrides to base configuration.

        Args:
            base_config: Base configuration dict.

        Returns:
            New config with overrides applied.
        """
        result = deep_copy_dict(base_config)
        apply_nested_overrides(result, self.overrides)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_id": self.variant_id,
            "variant_name": self.variant_name,
            "base_config_hash": self.base_config_hash,
            "overrides": self.overrides,
            "override_hash": self.override_hash,
            "unique_id": self.unique_id,
        }


@dataclass
class AblationMatrix:
    """
    Matrix of experiment configurations for ablation studies.

    Supports:
    - Single-value overrides: {"seed": 42}
    - List sweeps: {"temperature": [0.0, 0.5, 1.0]}
    - Cross-product expansion of all sweep values
    """

    name: str
    description: str = ""
    base_config: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Compute base config hash."""
        self._base_hash = compute_config_hash(self.base_config)

    @property
    def base_config_hash(self) -> str:
        """Hash of base configuration."""
        return self._base_hash

    def expand(self) -> list[AblationVariant]:
        """
        Expand matrix into list of variants.

        Each list value in overrides creates separate variants.

        Returns:
            List of AblationVariant instances.

        Example:
            >>> matrix = AblationMatrix(
            ...     name="temp_sweep",
            ...     base_config={"model": "llama"},
            ...     overrides={
            ...         "temperature": [0.0, 0.5, 1.0],
            ...         "seed": [42, 123],
            ...     },
            ... )
            >>> variants = matrix.expand()
            >>> len(variants)  # 3 temps * 2 seeds = 6
            6
        """
        # Separate sweep values from fixed overrides
        sweep_keys: list[str] = []
        sweep_values: list[list[Any]] = []
        fixed_overrides: dict[str, Any] = {}

        for key, value in self.overrides.items():
            if isinstance(value, list) and len(value) > 1:
                sweep_keys.append(key)
                sweep_values.append(value)
            else:
                # Single value or single-item list
                if isinstance(value, list) and len(value) == 1:
                    fixed_overrides[key] = value[0]
                else:
                    fixed_overrides[key] = value

        # Generate all combinations
        variants: list[AblationVariant] = []

        if sweep_values:
            for combo in itertools.product(*sweep_values):
                # Build override dict for this combination
                combo_overrides = dict(fixed_overrides)
                for i, key in enumerate(sweep_keys):
                    combo_overrides[key] = combo[i]

                # Generate variant name from key values
                name_parts = [f"{k}={v}" for k, v in zip(sweep_keys, combo)]
                variant_name = "_".join(name_parts) if name_parts else "base"

                variant = self._create_variant(
                    overrides=combo_overrides,
                    variant_name=variant_name,
                    variant_index=len(variants),
                )
                variants.append(variant)
        else:
            # No sweeps, just one variant with fixed overrides
            variant = self._create_variant(
                overrides=fixed_overrides,
                variant_name="base" if not fixed_overrides else "modified",
                variant_index=0,
            )
            variants.append(variant)

        return variants

    def _create_variant(
        self,
        overrides: dict[str, Any],
        variant_name: str,
        variant_index: int,
    ) -> AblationVariant:
        """Create a single variant."""
        override_hash = compute_config_hash(overrides)
        variant_id = f"{self.name}_{variant_index:03d}"

        return AblationVariant(
            variant_id=variant_id,
            variant_name=variant_name,
            base_config_hash=self.base_config_hash,
            overrides=overrides,
            override_hash=override_hash,
        )

    @classmethod
    def from_yaml(cls, path: str) -> AblationMatrix:
        """
        Load ablation matrix from YAML file.

        Expected format:
        ```yaml
        name: temperature_sweep
        description: Test different temperatures
        base_config:
          model: llama-7b
          seed: 42
        overrides:
          temperature: [0.0, 0.5, 1.0]
          top_p: 0.95
        ```

        Args:
            path: Path to YAML file.

        Returns:
            Loaded AblationMatrix.
        """
        import yaml
        from pathlib import Path

        content = Path(path).read_text()
        data = yaml.safe_load(content)

        return cls(
            name=data.get("name", "ablation"),
            description=data.get("description", ""),
            base_config=data.get("base_config", {}),
            overrides=data.get("overrides", {}),
        )

    def to_yaml(self, path: str) -> None:
        """
        Save ablation matrix to YAML file.

        Args:
            path: Path to write YAML.
        """
        import yaml
        from pathlib import Path

        data = {
            "name": self.name,
            "description": self.description,
            "base_config": self.base_config,
            "overrides": self.overrides,
        }

        content = yaml.dump(data, default_flow_style=False, sort_keys=True)
        Path(path).write_text(content)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "base_config_hash": self.base_config_hash,
            "overrides": self.overrides,
            "variant_count": len(self.expand()),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class AblationRun:
    """Tracks an ablation study execution."""

    matrix: AblationMatrix
    variants: list[AblationVariant] = field(default_factory=list)
    completed: dict[str, str] = field(default_factory=dict)  # variant_id -> run_id
    failed: dict[str, str] = field(default_factory=dict)  # variant_id -> error
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        """Expand matrix if not already done."""
        if not self.variants:
            self.variants = self.matrix.expand()

    @property
    def total_variants(self) -> int:
        """Total number of variants."""
        return len(self.variants)

    @property
    def completed_count(self) -> int:
        """Number of completed variants."""
        return len(self.completed)

    @property
    def failed_count(self) -> int:
        """Number of failed variants."""
        return len(self.failed)

    @property
    def pending_variants(self) -> list[AblationVariant]:
        """Variants that haven't been run yet."""
        done = set(self.completed.keys()) | set(self.failed.keys())
        return [v for v in self.variants if v.variant_id not in done]

    def mark_complete(self, variant_id: str, run_id: str) -> None:
        """Mark a variant as complete."""
        self.completed[variant_id] = run_id

    def mark_failed(self, variant_id: str, error: str) -> None:
        """Mark a variant as failed."""
        self.failed[variant_id] = error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "matrix": self.matrix.to_dict(),
            "variants": [v.to_dict() for v in self.variants],
            "completed": self.completed,
            "failed": self.failed,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_variants": self.total_variants,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
        }


def compute_config_hash(config: dict[str, Any]) -> str:
    """
    Compute stable hash of configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Hex-encoded hash.
    """
    json_bytes = canonical_json_bytes(config)
    return xxhash.xxh64(json_bytes).hexdigest()


def deep_copy_dict(d: dict[str, Any]) -> dict[str, Any]:
    """
    Deep copy a dictionary.

    Args:
        d: Dictionary to copy.

    Returns:
        Deep copy.
    """
    import copy

    return copy.deepcopy(d)


def apply_nested_overrides(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    """
    Apply nested overrides to target dict in-place.

    Supports dot notation for nested keys: "model.temperature" -> {"model": {"temperature": ...}}

    Args:
        target: Target dictionary to modify.
        overrides: Override values to apply.
    """
    for key, value in overrides.items():
        if "." in key:
            # Nested key
            parts = key.split(".")
            current = target

            # Navigate to parent
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set final value
            current[parts[-1]] = value
        else:
            # Simple key
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # Merge dicts recursively
                apply_nested_overrides(target[key], value)
            else:
                target[key] = value


def parse_override_string(override_str: str) -> dict[str, Any]:
    """
    Parse override string from CLI.

    Format: "key=value,key2=value2" or "key=value key2=value2"

    Args:
        override_str: Override string.

    Returns:
        Parsed overrides dict.

    Example:
        >>> parse_override_string("temperature=0.5,seed=42")
        {'temperature': 0.5, 'seed': 42}
    """
    import re

    overrides: dict[str, Any] = {}

    # Split by comma or whitespace
    pairs = re.split(r"[,\s]+", override_str.strip())

    for pair in pairs:
        if "=" not in pair:
            continue

        key, value_str = pair.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()

        # Try to parse value type
        value: Any
        if value_str.lower() in ("true", "false"):
            value = value_str.lower() == "true"
        elif value_str.isdigit():
            value = int(value_str)
        else:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str

        overrides[key] = value

    return overrides
