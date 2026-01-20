"""
Rubric Scoring Stage with Guided Decoding.

Evaluates content against scoring rubrics using LLMs with structured output.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from labelforge.pipelines.stage import Stage, StageConfig, StageContext, StageType

if TYPE_CHECKING:
    import ray.data

    from labelforge.core.prompt_pack import Rubric


class RubricScoreStage(Stage):
    """
    Stage that scores content using a rubric with guided decoding.

    This stage supports structured JSON output through vLLM's guided decoding
    to ensure consistent, parseable score outputs.

    Required inputs:
    - text: Text to score (or other content field)
    - row_id: Unique row identifier

    Outputs:
    - scores: Dict of criterion scores
    - overall_score: Aggregated score
    - score_reasoning: Model reasoning for scores
    - score_raw: Raw model output
    """

    def __init__(
        self,
        config: StageConfig,
        *,
        rubric: Rubric | None = None,
        text_field: str = "text",
        max_tokens: int = 512,
        temperature: float = 0.0,
        use_guided_decoding: bool = True,
    ):
        """
        Initialize rubric scoring stage.

        Args:
            config: Stage configuration.
            rubric: Scoring rubric to use.
            text_field: Name of field containing content to score.
            max_tokens: Maximum tokens in generated response.
            temperature: Sampling temperature (0.0 for deterministic).
            use_guided_decoding: Whether to use JSON schema-guided decoding.
        """
        super().__init__(config)
        self.rubric = rubric
        self.text_field = text_field
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_guided_decoding = use_guided_decoding

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
            "scores": dict,
            "overall_score": float,
            "score_reasoning": str,
            "score_raw": str,
        }

    def _build_rubric_prompt(self, content: str) -> str:
        """Build the rubric scoring prompt."""
        if self.rubric is None:
            return f"Score the following content:\n\n{content}"

        # Build detailed rubric prompt
        criteria_description = []
        for criterion in self.rubric.criteria:
            desc = f"- {criterion.name}: {criterion.description}"
            desc += f" (score {criterion.scale_min}-{criterion.scale_max})"
            if criterion.levels:
                levels_str = ", ".join(
                    f"{score}: {level}"
                    for score, level in sorted(criterion.levels.items())
                )
                desc += f"\n  Levels: {levels_str}"
            criteria_description.append(desc)

        criteria_text = "\n".join(criteria_description)

        prompt = f"""Score the following content using the rubric below.

## Rubric: {self.rubric.name}
{self.rubric.description}

## Criteria:
{criteria_text}

## Content to Score:
{content}

## Instructions:
Provide your scores as a JSON object with:
- "scores": {{criterion_name: score, ...}} for each criterion
- "overall_score": weighted average score
- "reasoning": brief explanation for your scores

Respond ONLY with valid JSON."""

        return prompt

    def _get_output_schema(self) -> dict[str, Any]:
        """Get JSON schema for guided decoding."""
        if self.rubric is None:
            properties = {
                "scores": {"type": "object"},
                "overall_score": {"type": "number"},
                "reasoning": {"type": "string"},
            }
        else:
            # Build schema with specific criteria
            score_properties = {}
            for criterion in self.rubric.criteria:
                score_properties[criterion.name] = {
                    "type": "number",
                    "minimum": criterion.scale_min,
                    "maximum": criterion.scale_max,
                }

            properties = {
                "scores": {
                    "type": "object",
                    "properties": score_properties,
                    "required": list(score_properties.keys()),
                },
                "overall_score": {
                    "type": "number",
                    "minimum": self.rubric.overall_scale_min,
                    "maximum": self.rubric.overall_scale_max,
                },
                "reasoning": {"type": "string"},
            }

        return {
            "type": "object",
            "properties": properties,
            "required": ["scores", "overall_score", "reasoning"],
        }

    def _create_preprocess_fn(
        self,
        context: StageContext,
    ) -> Any:
        """Create preprocessing function for Ray Data LLM."""
        text_field = self.text_field
        max_tokens = self.max_tokens
        temperature = self.temperature
        use_guided_decoding = self.use_guided_decoding
        rubric_prompt_builder = self._build_rubric_prompt
        output_schema = self._get_output_schema()

        def preprocess(row: dict[str, Any]) -> dict[str, Any]:
            """Prepare row for rubric scoring."""
            content = row.get(text_field, "")

            # Build rubric prompt
            prompt = rubric_prompt_builder(content)

            # Create messages
            messages = [{"role": "user", "content": prompt}]

            sampling_params: dict[str, Any] = {
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            # Add guided decoding if enabled
            if use_guided_decoding:
                sampling_params["guided_decoding"] = {
                    "json": output_schema,
                }

            return {
                "messages": messages,
                "sampling_params": sampling_params,
                "row_id": row.get("row_id", ""),
            }

        return preprocess

    def _create_postprocess_fn(self) -> Any:
        """Create postprocessing function for Ray Data LLM."""
        rubric = self.rubric

        def postprocess(row: dict[str, Any]) -> dict[str, Any]:
            """Process rubric scoring output."""
            generated_text = row.get("generated_text", "")

            result: dict[str, Any] = {
                "row_id": row.get("row_id", ""),
                "score_raw": generated_text,
                "scores": {},
                "overall_score": 0.0,
                "score_reasoning": "",
            }

            # Try to parse JSON output
            try:
                parsed = json.loads(generated_text.strip())
                result["scores"] = parsed.get("scores", {})
                result["overall_score"] = float(parsed.get("overall_score", 0.0))
                result["score_reasoning"] = parsed.get("reasoning", "")
            except (json.JSONDecodeError, ValueError, TypeError):
                result["parse_error"] = True
                # Fallback: try to compute weighted average if we have partial scores
                if rubric and result["scores"]:
                    total_weight = sum(c.weight for c in rubric.criteria)
                    weighted_sum = sum(
                        result["scores"].get(c.name, 0) * c.weight
                        for c in rubric.criteria
                    )
                    if total_weight > 0:
                        result["overall_score"] = weighted_sum / total_weight

            return result

        return postprocess

    def run(
        self,
        dataset: ray.data.Dataset,
        context: StageContext,
    ) -> ray.data.Dataset:
        """
        Execute the rubric scoring stage on a dataset.

        Args:
            dataset: Input Ray Dataset.
            context: Execution context.

        Returns:
            Dataset with added score fields.
        """
        from labelforge.core.model_spec import ModelSpec
        from labelforge.llm.processor_factory import apply_processor_to_dataset

        # Create model spec
        model_spec_name = self.config.model_spec_name
        if model_spec_name:
            model_spec = ModelSpec(model_source=model_spec_name)
        else:
            model_spec = ModelSpec(
                model_source="meta-llama/Llama-3.1-8B-Instruct",
                task_type="generate",
            )

        # Create preprocessing and postprocessing functions
        preprocess_fn = self._create_preprocess_fn(context)
        postprocess_fn = self._create_postprocess_fn()

        # Apply processor to dataset
        return apply_processor_to_dataset(
            dataset=dataset,
            model_spec=model_spec,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
        )


def create_rubric_score_stage(
    name: str = "rubric_score",
    *,
    rubric: Rubric | None = None,
    text_field: str = "text",
    max_tokens: int = 512,
    temperature: float = 0.0,
    use_guided_decoding: bool = True,
    model_spec_name: str | None = None,
    version: str = "1.0.0",
) -> RubricScoreStage:
    """
    Create a rubric scoring stage with configuration.

    Args:
        name: Stage name.
        rubric: Scoring rubric to use.
        text_field: Name of field containing content to score.
        max_tokens: Maximum tokens in generated response.
        temperature: Sampling temperature.
        use_guided_decoding: Whether to use JSON schema-guided decoding.
        model_spec_name: Optional model specification name.
        version: Stage version.

    Returns:
        Configured RubricScoreStage.

    Example:
        >>> from labelforge.core.prompt_pack import Rubric, RubricCriterion
        >>> rubric = Rubric(
        ...     name="quality",
        ...     version="1.0",
        ...     description="Text quality rubric",
        ...     criteria=[
        ...         RubricCriterion(
        ...             name="clarity",
        ...             description="How clear is the text",
        ...         ),
        ...     ],
        ... )
        >>> stage = create_rubric_score_stage(name="scorer", rubric=rubric)
        >>> stage.name
        'scorer'
    """
    config = StageConfig(
        name=name,
        stage_type=StageType.RUBRIC_SCORE,
        version=version,
        model_spec_name=model_spec_name,
        params={
            "text_field": text_field,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_guided_decoding": use_guided_decoding,
        },
    )

    return RubricScoreStage(
        config=config,
        rubric=rubric,
        text_field=text_field,
        max_tokens=max_tokens,
        temperature=temperature,
        use_guided_decoding=use_guided_decoding,
    )
