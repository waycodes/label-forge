# Run Contract

This document defines what constitutes a "run" and the invariants for replay.

## Definition of a Run

A **run** is a complete execution of a LabelForge pipeline defined by:

| Component | Description |
|-----------|-------------|
| **Code Revision** | Git commit hash of the codebase |
| **Pipeline Config** | YAML configuration specifying stages and dependencies |
| **Prompt Pack** | Versioned prompt templates, rubrics, and schemas |
| **Model Spec** | Model identifiers, revisions, and engine configuration |
| **Seeds** | Run seed plus derived stage and row seeds |
| **Hardware Profile** | GPU type, count, and CUDA version |

## Run Identifier

Each run is assigned a unique `run_id` (8-character UUID prefix) that serves as:
- Directory name for outputs
- Key in manifest references
- Prefix for cache entries

## Output Structure

```
runs/<run_id>/
├── manifest.json           # Run-level manifest
├── env_snapshot.json       # Environment capture
├── stages/
│   ├── <stage_name>/
│   │   ├── manifest.json   # Stage manifest
│   │   ├── rows.jsonl      # Row-level records
│   │   └── output/         # Stage output data
│   └── ...
├── logs/
│   └── run.log
└── reports/
    └── summary.md
```

## Replay Invariants

### MUST Match (Strict)
For a replay to be considered valid, these must be identical:

1. **Manifest Hash** - Overall fingerprint of run configuration
2. **Row Counts** - Same number of input/output rows per stage
3. **Schema Fingerprints** - Output schemas match expected

### SHOULD Match (Best Effort)
These should match but may have known tolerances:

1. **Per-Row Cache Keys** - Stable given same inputs
2. **Output Content** - Byte-identical under deterministic mode

### Known Non-Invariants
These are NOT expected to match between runs:

1. **Timestamps** - Execution times will differ
2. **Absolute Paths** - May differ between machines
3. **Latency Metrics** - Performance varies by load

## Determinism Modes

### Standard Mode (Default)
- `VLLM_BATCH_INVARIANT=1`
- Outputs are batch-size independent
- Best throughput with reproducible results

### Strict Mode
- `VLLM_ENABLE_V1_MULTIPROCESSING=0`
- Ray Data `preserve_order=True`
- Maximum reproducibility, reduced throughput

## Verification

### Cache Key Stability Test
```bash
# Run same config twice
labelforge run --config configs/test.yaml --seed 42 --output run1
labelforge run --config configs/test.yaml --seed 42 --output run2

# Compare cache key coverage
labelforge diff run1 run2 --cache-keys
```

### Manifest Hash Test
```python
from labelforge.core.manifest import RunManifest, compute_manifest_hash

m1 = RunManifest.load("run1/manifest.json")
m2 = RunManifest.load("run2/manifest.json")

assert compute_manifest_hash(m1) == compute_manifest_hash(m2)
```

## Requirements for Reproducibility

| Requirement | Notes |
|-------------|-------|
| Same vLLM version | Different versions may produce different outputs |
| Same GPU type | Different architectures affect precision |
| Pinned model revision | Use HuggingFace commit hashes |
| Fixed seeds | Explicit in config |
| Determinism mode enabled | Set appropriate env vars |
