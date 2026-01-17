# Determinism Caveats

LabelForge strives for reproducible outputs, but there are important caveats.

## vLLM Reproducibility

> **From vLLM docs**: Reproducibility requires same hardware and vLLM version.

### Environment Variables

| Variable | Effect |
|----------|--------|
| `VLLM_BATCH_INVARIANT=1` | Outputs independent of batching |
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | Disable async multiprocessing (stricter) |

### What "Batch Invariant" Means

With `VLLM_BATCH_INVARIANT=1`:
- Batching the same N requests differently → same outputs
- Does NOT mean different batch sizes → same outputs

### Hardware Constraints

- **GPU Type**: A100 vs H100 may produce different results
- **GPU Count**: Tensor parallelism affects numerical precision
- **Driver Version**: May affect CUDA behavior

## Ray Data Ordering

> **From Ray docs**: Dataset iteration order is not guaranteed unless `preserve_order=True`.

### Without `preserve_order`
```python
# Tasks may complete in any order
ds.map(fn)  # Order not guaranteed
```

### With `preserve_order`
```python
# Enable ordering
ctx = ray.data.DataContext.get_current()
ctx.execution_options.preserve_order = True
```

### LabelForge Approach

LabelForge does NOT rely on ordering for correctness:
- Every row has a stable `row_id`
- All outputs keyed by `row_id`
- Manifests are sorted by `row_id`

## Known Sources of Non-Determinism

### Model-Level
1. **Attention computation**: Floating-point associativity
2. **Flash Attention**: Approximate algorithms
3. **Quantization**: Different precision effects

### System-Level
1. **CUDA rounding**: GPU-specific behavior
2. **Thread scheduling**: Affects float accumulation
3. **Memory allocation**: May affect computation order

### Ray-Level
1. **Task scheduling**: Non-deterministic without preserve_order
2. **Actor placement**: Different resource allocation
3. **Autoscaling**: Dynamic worker count

## Best Practices

### For Maximum Reproducibility

```yaml
# config.yaml
determinism:
  mode: strict
  seed: 42
  vllm_batch_invariant: true
  vllm_multiprocessing: false
  ray_preserve_order: true
```

### For Production Throughput

```yaml
# config.yaml
determinism:
  mode: standard
  seed: 42
  vllm_batch_invariant: true
  # Allow multiprocessing and async execution
```

## Verifying Reproducibility

### Quick Check
```bash
# Run twice and compare
labelforge run --config test.yaml --seed 42 -o run1
labelforge run --config test.yaml --seed 42 -o run2
labelforge diff run1 run2
```

### Expected Outcomes

| Component | Standard Mode | Strict Mode |
|-----------|--------------|-------------|
| Manifest hash | ✅ Match | ✅ Match |
| Row counts | ✅ Match | ✅ Match |
| Output values | ⚠️ May vary | ✅ High match |
| Execution order | ❌ May differ | ✅ Preserved |

## Escape Hatches

When you CAN'T guarantee reproducibility:

1. **Document known variance** in run notes
2. **Use row_id tracking** for partial replay
3. **Accept bounded tolerance** for certain outputs
4. **Version-lock all dependencies** via container

## Further Reading

- [vLLM Reproducibility](https://docs.vllm.ai/en/latest/usage/reproducibility/)
- [Ray Data Shuffling](https://docs.ray.io/en/latest/data/shuffling-data.html)
