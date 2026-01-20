"""
Text Embedding Stage.

Generates embeddings for text using vLLM pooling models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from labelforge.pipelines.stage import Stage, StageConfig, StageContext, StageType

if TYPE_CHECKING:
    import ray.data


class TextEmbeddingStage(Stage):
    """
    Stage that generates text embeddings using an embedding model.

    Uses vLLM's pooling/embedding models via Ray Data LLM.

    Required inputs:
    - text: Text content to embed
    - row_id: Unique row identifier

    Outputs:
    - embedding: Dense vector embedding
    - embedding_model: Model name for provenance
    """

    def __init__(
        self,
        config: StageConfig,
        *,
        text_field: str = "text",
        embedding_dim: int | None = None,
        normalize: bool = True,
        dtype: str = "float32",
    ):
        """
        Initialize text embedding stage.

        Args:
            config: Stage configuration.
            text_field: Name of field containing text to embed.
            embedding_dim: Expected embedding dimension (for validation).
            normalize: Whether to L2-normalize embeddings.
            dtype: Data type for embeddings.
        """
        super().__init__(config)
        self.text_field = text_field
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.dtype = dtype

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
            "embedding": list,
            "embedding_model": str,
            "embedding_dim": int,
        }

    def _create_preprocess_fn(self, context: StageContext) -> Any:
        """Create preprocessing function for Ray Data LLM."""
        text_field = self.text_field

        def preprocess(row: dict[str, Any]) -> dict[str, Any]:
            """Prepare row for embedding."""
            text = row.get(text_field, "")

            # For embedding models, we just pass the text directly
            return {
                "prompt": text,
                "row_id": row.get("row_id", ""),
            }

        return preprocess

    def _create_postprocess_fn(self) -> Any:
        """Create postprocessing function for Ray Data LLM."""
        normalize = self.normalize
        model_name = self.config.model_spec_name or "unknown"

        def postprocess(row: dict[str, Any]) -> dict[str, Any]:
            """Process embedding output."""
            embedding = row.get("embedding", [])

            # Convert to list if needed
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()

            # Normalize if requested
            if normalize and embedding:
                import math

                norm = math.sqrt(sum(x * x for x in embedding))
                if norm > 0:
                    embedding = [x / norm for x in embedding]

            return {
                "row_id": row.get("row_id", ""),
                "embedding": embedding,
                "embedding_model": model_name,
                "embedding_dim": len(embedding) if embedding else 0,
            }

        return postprocess

    def run(
        self,
        dataset: ray.data.Dataset,
        context: StageContext,
    ) -> ray.data.Dataset:
        """
        Execute the embedding stage on a dataset.

        Args:
            dataset: Input Ray Dataset with text.
            context: Execution context.

        Returns:
            Dataset with added embedding fields.
        """
        from labelforge.core.model_spec import ModelSpec
        from labelforge.llm.processor_factory import apply_processor_to_dataset

        # Create model spec for embedding
        model_spec_name = self.config.model_spec_name
        if model_spec_name:
            model_spec = ModelSpec(
                model_source=model_spec_name,
                task_type="embed",
            )
        else:
            # Default to a sentence transformer style model
            model_spec = ModelSpec(
                model_source="sentence-transformers/all-MiniLM-L6-v2",
                task_type="embed",
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


def create_text_embedding_stage(
    name: str = "embed_text",
    *,
    text_field: str = "text",
    embedding_dim: int | None = None,
    normalize: bool = True,
    model_spec_name: str | None = None,
    version: str = "1.0.0",
) -> TextEmbeddingStage:
    """
    Create a text embedding stage.

    Args:
        name: Stage name.
        text_field: Name of field containing text.
        embedding_dim: Expected embedding dimension.
        normalize: Whether to L2-normalize embeddings.
        model_spec_name: Optional embedding model name.
        version: Stage version.

    Returns:
        Configured TextEmbeddingStage.

    Example:
        >>> stage = create_text_embedding_stage(
        ...     name="caption_embeddings",
        ...     text_field="caption",
        ...     model_spec_name="sentence-transformers/all-mpnet-base-v2",
        ... )
        >>> stage.name
        'caption_embeddings'
    """
    config = StageConfig(
        name=name,
        stage_type=StageType.EMBED,
        version=version,
        model_spec_name=model_spec_name,
        params={
            "text_field": text_field,
            "embedding_dim": embedding_dim,
            "normalize": normalize,
        },
    )

    return TextEmbeddingStage(
        config=config,
        text_field=text_field,
        embedding_dim=embedding_dim,
        normalize=normalize,
    )
