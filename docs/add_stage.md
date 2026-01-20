# How to Add a New Stage

This guide walks through creating a new pipeline stage in LabelForge.

## Overview

Stages are the building blocks of LabelForge pipelines. Each stage:
- Transforms an input dataset into an output dataset
- Has a defined input and output schema
- Produces deterministic, cacheable results
- Integrates with the manifest and caching systems

## Step 1: Create the Stage Class

Create a new file in `labelforge/pipelines/stages/`:

```python
# labelforge/pipelines/stages/sentiment.py

from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any

from labelforge.pipelines.stage import Stage, StageConfig, StageContext, StageType

if TYPE_CHECKING:
    import ray.data


class SentimentStage(Stage):
    """Analyze sentiment of text content."""

    def __init__(
        self,
        config: StageConfig,
        *,
        text_field: str = "text",
        max_tokens: int = 128,
        temperature: float = 0.0,
    ):
        super().__init__(config)
        self.text_field = text_field
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def input_schema(self) -> dict[str, type]:
        """Required input fields."""
        return {
            self.text_field: str,
            "row_id": str,
        }

    @property
    def output_schema(self) -> dict[str, type]:
        """Output fields added by this stage."""
        return {
            "sentiment": str,
            "confidence": float,
        }
```

## Step 2: Implement Preprocessing

The preprocessing function converts input rows to LLM-ready format:

```python
def _create_preprocess_fn(self, context: StageContext) -> Any:
    """Create preprocessing function for Ray Data LLM."""
    text_field = self.text_field
    max_tokens = self.max_tokens
    temperature = self.temperature

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        text = row.get(text_field, "")
        
        messages = [
            {"role": "system", "content": "Analyze sentiment. Return JSON."},
            {"role": "user", "content": f"Text: {text}"},
        ]

        return {
            "messages": messages,
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "guided_decoding": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["sentiment", "confidence"]
                    }
                }
            },
            "row_id": row.get("row_id", ""),
        }

    return preprocess
```

## Step 3: Implement Postprocessing

The postprocessing function handles model outputs:

```python
def _create_postprocess_fn(self) -> Any:
    """Create postprocessing function."""
    
    def postprocess(row: dict[str, Any]) -> dict[str, Any]:
        generated_text = row.get("generated_text", "")
        
        result = {
            "row_id": row.get("row_id", ""),
            "sentiment": "neutral",
            "confidence": 0.0,
        }

        try:
            parsed = json.loads(generated_text.strip())
            result["sentiment"] = parsed.get("sentiment", "neutral")
            result["confidence"] = float(parsed.get("confidence", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            result["parse_error"] = True

        return result

    return postprocess
```

## Step 4: Implement the Run Method

Wire everything together:

```python
def run(
    self,
    dataset: ray.data.Dataset,
    context: StageContext,
) -> ray.data.Dataset:
    """Execute the stage."""
    from labelforge.core.model_spec import ModelSpec
    from labelforge.llm.processor_factory import apply_processor_to_dataset

    model_spec = ModelSpec(
        model_source=self.config.model_spec_name or "meta-llama/Llama-3.1-8B-Instruct",
        task_type="generate",
    )

    return apply_processor_to_dataset(
        dataset=dataset,
        model_spec=model_spec,
        preprocess_fn=self._create_preprocess_fn(context),
        postprocess_fn=self._create_postprocess_fn(),
    )
```

## Step 5: Create a Factory Function

Add a convenient factory function:

```python
def create_sentiment_stage(
    name: str = "sentiment",
    *,
    text_field: str = "text",
    max_tokens: int = 128,
    temperature: float = 0.0,
    model_spec_name: str | None = None,
    version: str = "1.0.0",
) -> SentimentStage:
    """Create a sentiment analysis stage."""
    config = StageConfig(
        name=name,
        stage_type=StageType.TEXT_LABEL,  # Reuse existing type
        version=version,
        model_spec_name=model_spec_name,
        params={
            "text_field": text_field,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    return SentimentStage(
        config=config,
        text_field=text_field,
        max_tokens=max_tokens,
        temperature=temperature,
    )
```

## Step 6: Register in Module Exports

Update `labelforge/pipelines/stages/__init__.py`:

```python
from labelforge.pipelines.stages.sentiment import (
    SentimentStage,
    create_sentiment_stage,
)

__all__ = [
    # ... existing exports
    "SentimentStage",
    "create_sentiment_stage",
]
```

## Step 7: Write Tests

Create `tests/test_sentiment_stage.py`:

```python
import pytest
from labelforge.pipelines.stages.sentiment import create_sentiment_stage


class TestSentimentStage:
    """Tests for SentimentStage."""

    def test_create_stage(self) -> None:
        stage = create_sentiment_stage()
        assert stage.name == "sentiment"
        assert "text" in stage.input_schema
        assert "sentiment" in stage.output_schema

    def test_postprocess_valid(self) -> None:
        stage = create_sentiment_stage()
        postprocess = stage._create_postprocess_fn()
        
        result = postprocess({
            "row_id": "1",
            "generated_text": '{"sentiment": "positive", "confidence": 0.95}'
        })
        
        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.95

    def test_fingerprint_stability(self) -> None:
        s1 = create_sentiment_stage(name="test")
        s2 = create_sentiment_stage(name="test")
        assert s1.fingerprint() == s2.fingerprint()
```

## Using Your Stage

Register and use in a pipeline:

```python
from labelforge.pipelines.dag import PipelineDAG
from labelforge.pipelines.runner import create_runner, RunConfig
from labelforge.pipelines.stages.sentiment import create_sentiment_stage

# Create DAG
dag = PipelineDAG()
dag.add_node("sentiment_analysis", stage_type="text_label")

# Create and register stage
sentiment_stage = create_sentiment_stage(
    name="sentiment_analysis",
    text_field="review_text",
)

# Create runner
runner = create_runner(dag, RunConfig(seed=42))
runner.register_stage(sentiment_stage)

# Execute
results = runner.run(input_dataset)
```

## Best Practices

1. **Determinism**: Use `temperature=0.0` for reproducible outputs
2. **Guided Decoding**: Use JSON schemas to enforce output structure
3. **Error Handling**: Always handle JSON parse failures gracefully
4. **Fingerprinting**: Include all configuration in stage params for cache invalidation
5. **Testing**: Test preprocessing, postprocessing, and fingerprint stability
