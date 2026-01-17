# I/O Layout

Standard directory structure for LabelForge artifacts.

## Run Artifacts

```
runs/
└── <run_id>/
    ├── manifest.json           # Run-level manifest
    ├── env_snapshot.json       # Environment capture at run start
    ├── config.yaml             # Copy of run configuration
    │
    ├── stages/                 # Per-stage outputs
    │   └── <stage_name>/
    │       ├── manifest.json   # Stage manifest
    │       ├── rows.jsonl      # Row-level records
    │       └── output/
    │           ├── part-0000.parquet
    │           ├── part-0001.parquet
    │           └── _metadata
    │
    ├── logs/
    │   ├── run.log             # Main run log
    │   └── stages/
    │       └── <stage_name>.log
    │
    ├── metrics/
    │   └── metrics.json        # Aggregated metrics
    │
    └── reports/
        └── summary.md          # Human-readable summary
```

## Cache Layout

```
.cache/
├── metadata.db             # SQLite metadata store
└── blobs/
    └── <prefix[0:2]>/
        └── <prefix[2:4]>/
            └── <hash>.json.gz
```

## Prompts Layout

```
prompts/
└── <pack_name>/
    ├── pack.yaml           # Pack definition
    ├── templates/          # Optional subdirectories
    │   └── caption.yaml
    └── rubrics/
        └── quality.yaml
```

## Configurations

```
configs/
├── mvp.yaml                # MVP pipeline config
├── engines/
│   ├── vllm_throughput.yaml
│   └── vllm_deterministic.yaml
├── presets/
│   ├── dev_small.yaml
│   └── production.yaml
└── runtime_env/
    ├── local.yaml
    └── cluster.yaml
```

## Output Formats

### Parquet (Default)
- Schema-evolution friendly
- Efficient columnar storage
- Native Ray Data support

### JSONL (Manifests)
- Human-readable
- Append-only writes
- Stream processing

## Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Run ID | `<uuid-prefix>` | `a1b2c3d4` |
| Stage output | `<stage_name>/output/` | `caption/output/` |
| Parquet parts | `part-NNNN.parquet` | `part-0000.parquet` |
| Cache blobs | `<hash>.json.gz` | `1a2b3c4d5e6f7g8h.json.gz` |
