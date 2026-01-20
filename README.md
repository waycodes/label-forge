# LabelForge

**Deterministic, replayable, ablation-friendly multimodal labeling and synthetic-data pipeline built on Ray + vLLM.**

## Features

- **Multimodal Labeling**: VLM-based image captioning, attribute tagging, and text classification
- **Deterministic Pipelines**: Prompt/model pinning, stable row IDs, and explicit seed management
- **Replayable Runs**: Content-addressed caching and comprehensive manifests for full reproducibility
- **Ablation-Friendly**: Matrix experiments with shared caches and run comparison tools
- **Rubric Scoring**: Large-batch structured evaluation with guided decoding
- **Hard-Negative Mining**: Embedding + reranking pipeline for high-quality training data
- **Synthetic Data Generation**: Text and VLM-grounded conversation synthesis with quality filtering

## Architecture

LabelForge uses **Ray Data LLM** (`build_processor` + `vLLMEngineProcessorConfig`) as the inference backbone, providing:

- Efficient batch inference across multiple GPUs
- Per-row sampling parameters for ablations
- Native VLM support with PIL image inputs
- Guided decoding for structured JSON outputs

### Key Components

```
labelforge/
├── core/           # Schemas, hashing, seeds, environment capture
├── io/             # Dataset I/O, JSONL manifests, images
├── llm/            # Ray Data LLM processor factory, determinism toggles
├── pipelines/      # Stage abstraction, DAG, runner
├── cache/          # Content-addressed row/stage caching
├── mining/         # Hard-negative candidate generation and selection
├── synth/          # Synthetic data generation and deduplication
├── eval/           # Score normalization and metrics
└── cli/            # Command-line interface
```

## Installation

```bash
# Basic installation
pip install labelforge

# With S3 cache backend
pip install labelforge[s3]

# Development
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run a labeling pipeline
labelforge run --config configs/mvp.yaml

# Replay a previous run
labelforge replay --manifest runs/<run_id>/manifest.jsonl

# Compare two runs
labelforge diff runs/<run_a> runs/<run_b>

# Inspect run artifacts
labelforge inspect runs/<run_id>
```

## Determinism

LabelForge provides two determinism modes:

1. **Standard Mode** (default): Uses `VLLM_BATCH_INVARIANT=1` for scheduling-insensitive outputs. Best throughput with reproducible results.

2. **Strict Mode**: Additionally enables Ray Data `preserve_order` and disables vLLM multiprocessing. Maximum reproducibility at the cost of throughput.

### Requirements for Reproducibility

- Same code revision and config
- Pinned prompt pack version
- Pinned model revision
- Fixed seeds
- Same hardware profile (GPU type, count)
- Same Ray + vLLM versions

See [docs/determinism.md](docs/determinism.md) for detailed caveats.

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Run Contract](docs/run_contract.md)
- [I/O Layout](docs/io_layout.md)
- [Determinism Caveats](docs/determinism.md)
- [Adding New Stages](docs/add_stage.md)
- [Running Ablations](docs/ablations.md)
- [Performance Tuning](docs/perf_tuning.md)

## License and Usage

This software is **proprietary** under a Portfolio/Research-Only License.

- **No commercial or professional use** permitted
- **Research use requires citation** — see [CITATION.cff](CITATION.cff)
- **No external contributions** accepted

See [LICENSE](LICENSE) for full terms.
