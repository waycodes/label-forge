"""
Manifest hashing for run equivalence checks.

Computes stable hashes of manifest content for comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import xxhash

from labelforge.core.json_canonical import canonical_json_bytes

if TYPE_CHECKING:
    from labelforge.core.manifest.run_manifest import RunManifest


def compute_manifest_hash(manifest: RunManifest) -> str:
    """
    Compute hash of a run manifest.

    The hash excludes volatile fields (timestamps, paths) and focuses
    on content that affects reproducibility.

    Args:
        manifest: The run manifest to hash.

    Returns:
        Hex-encoded hash string.
    """
    # Extract reproducibility-relevant fields
    content = {
        "run_seed": manifest.metadata.run_seed,
        "git_commit": manifest.metadata.git_commit,
        "config_hash": manifest.metadata.config_hash,
        "prompt_packs": manifest.prompt_packs,
        "model_specs": manifest.model_specs,
        "stages": [
            {
                "stage_name": s.stage_name,
                "stage_type": s.stage_type,
                "stage_version": s.stage_version,
                "stage_hash": s.stage_hash,
                "depends_on": s.depends_on,
            }
            for s in manifest.stages
        ],
    }

    json_bytes = canonical_json_bytes(content)
    return xxhash.xxh64(json_bytes).hexdigest()


def compute_file_hash(path: Path) -> str:
    """
    Compute hash of a file's contents.

    Args:
        path: Path to file.

    Returns:
        Hex-encoded hash string.
    """
    hasher = xxhash.xxh64()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_jsonl_content_hash(path: Path) -> str:
    """
    Compute hash of JSONL file content (line-order independent).

    This is useful for comparing manifests where row order may vary
    but content should be identical.

    Args:
        path: Path to JSONL file.

    Returns:
        Hex-encoded hash string.
    """
    from labelforge.core.json_canonical import canonical_json_bytes, canonical_json_loads

    # Read all lines, parse, and sort by a stable key
    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                record = canonical_json_loads(line)
                lines.append(record)

    # Sort by row_id if present, otherwise use full content
    if lines and "row_id" in lines[0]:
        lines.sort(key=lambda r: r.get("row_id", ""))
    else:
        lines.sort(key=lambda r: canonical_json_bytes(r))

    # Hash sorted content
    combined = canonical_json_bytes(lines)
    return xxhash.xxh64(combined).hexdigest()


def compare_manifests(
    manifest_a: RunManifest,
    manifest_b: RunManifest,
) -> dict[str, bool]:
    """
    Compare two run manifests for equivalence.

    Args:
        manifest_a: First manifest.
        manifest_b: Second manifest.

    Returns:
        Dict of comparison results by component.
    """
    return {
        "run_seed_match": manifest_a.metadata.run_seed == manifest_b.metadata.run_seed,
        "git_commit_match": manifest_a.metadata.git_commit
        == manifest_b.metadata.git_commit,
        "config_hash_match": manifest_a.metadata.config_hash
        == manifest_b.metadata.config_hash,
        "prompt_packs_match": manifest_a.prompt_packs == manifest_b.prompt_packs,
        "model_specs_match": manifest_a.model_specs == manifest_b.model_specs,
        "stage_count_match": len(manifest_a.stages) == len(manifest_b.stages),
        "overall_hash_match": compute_manifest_hash(manifest_a)
        == compute_manifest_hash(manifest_b),
    }
