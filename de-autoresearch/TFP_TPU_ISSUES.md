# TensorFlow Probability TPU Performance Issues in GaussianNetwork

## Executive Summary

The use of TensorFlow Probability (TFP) `DistributionLambda` layers and `Distribution` objects in `GaussianNetwork` output layers creates significant performance bottlenecks on TPU that don't affect GPU performance as severely. These issues stem from XLA compilation challenges, Python-level abstractions, and non-vectorized operations.

## Critical Issues

### 1. **DistributionLambda with Lambda Functions (CRITICAL)**

**Location**: `trieste/models/keras/architectures.py:340-353`

```python
def distribution_fn(inputs: TensorType) -> tfp.distributions.Distribution:
    return tfp.distributions.Normal(inputs[..., :1], tf.math.softplus(inputs[..., 1:]))

distribution = tfp.layers.DistributionLambda(
    make_distribution_fn=distribution_fn,
    convert_to_tensor_fn=tfp.distributions.Distribution.mean,
    ...
)(parameter_layer)
```

**The Problem**:
- `DistributionLambda` wraps a **Python lambda function** (`distribution_fn`)
- Lambda functions create **complex control flow** in the computation graph
- XLA compiler struggles to optimize lambda-based layers
- The `convert_to_tensor_fn` adds another layer of abstraction

**TPU Impact**:
- **Poor XLA compilation**: Lambda functions don't compile efficiently to HLO (High-Level Optimizer) graphs
- **Graph bloat**: XLA must trace through entire TFP distribution logic
- **Inefficient kernels**: Results in suboptimal TPU kernel generation
- **GPU advantage**: GPUs handle this better due to lower compilation overhead

**Why TPU Struggles**:
- TPUs require **static, well-optimized graphs** for efficient execution
- Lambda functions introduce dynamic behavior that XLA can't fully optimize
- TPU's systolic array architecture needs predictable, fused operations

**Recommendation**:
- Replace `DistributionLambda` with **direct tensor operations**
- Output `[mean, log_variance]` concatenated tensor instead of Distribution object
- Compute loss manually using pure TensorFlow operations

### 2. **Distribution Objects as Model Outputs (CRITICAL)**

**Location**: `trieste/models/keras/architectures.py:331-336` (multi-output), `trieste/models/keras/models.py:251-260`

```python
# Model returns Distribution objects
ensemble_distributions = self.ensemble_distributions(flat_x)  # Returns tuple of Distributions

# Then extracts mean/variance in Python loop
predicted_means = tf.stack(
    [unflatten(dist.mean()) for dist in ensemble_distributions], axis=-3
)
```

**The Problem**:
- Model outputs are **TFP Distribution objects**, not tensors
- `Distribution` is a high-level Python abstraction
- Extracting statistics requires **Python-level method calls** (`.mean()`, `.variance()`)
- These operations happen **outside the compiled graph**

**TPU Impact**:
- **CPU-TPU synchronization**: Distribution method calls may execute on CPU
- **No vectorization**: Python list comprehension processes distributions sequentially
- **Graph breaks**: Operations outside compiled graph can't be optimized by XLA
- **Memory overhead**: Distribution objects carry metadata and validation logic

**Why This Hurts TPU**:
- TPU excels when **everything is in the compiled graph**
- Python-level operations break the graph, forcing CPU execution
- Each `.mean()` call may trigger separate TPU kernel launch
- No batching/vectorization across ensemble members

**Recommendation**:
- Model should output **raw tensors**: `[mean, variance]` or `[mean, log_variance]`
- All operations should be **pure TensorFlow operations** within the graph
- Use `tf.stack` and `tf.reduce_mean` for ensemble aggregation (vectorized)

### 3. **Log-Probability Computation in Loss Function (HIGH)**

**Location**: `trieste/models/keras/utils.py:125-136`, `trieste/models/keras/models.py:158`

```python
def negative_log_likelihood(
    y_true: TensorType, y_pred: tfp.distributions.Distribution
) -> TensorType:
    return -y_pred.log_prob(y_true)
```

**The Problem**:
- Loss function calls `dist.log_prob(y_true)` on TFP Distribution
- TFP's `log_prob` includes:
  - Shape validation and broadcasting logic
  - Multiple internal transformations
  - Complex control flow for different distribution types
- This happens **during training** for every batch

**TPU Impact**:
- **Complex graph**: XLA must compile entire TFP log_prob logic
- **Inefficient kernels**: Results in many small operations instead of fused kernel
- **Broadcasting overhead**: TFP's broadcasting may not align with TPU memory requirements
- **No vectorization**: Each ensemble member's loss computed separately

**Why TPU Struggles**:
- TPU prefers **simple, fused operations** (like single matrix multiply)
- TFP's log_prob creates complex graph with many small operations
- GPU handles this better due to more flexible execution model

**Recommendation**:
- Implement **manual Gaussian NLL** using pure TensorFlow:
  ```python
  def gaussian_nll(y_true, y_pred):
      mean, log_var = tf.split(y_pred, 2, axis=-1)
      precision = tf.exp(-log_var)
      return tf.reduce_mean(0.5 * precision * (y_true - mean)**2 + 0.5 * log_var)
  ```
- This compiles to **single efficient TPU kernel**
- Can be vectorized across ensemble members

### 4. **Python Loops Over Distributions (HIGH)**

**Location**: `trieste/models/keras/models.py:385-391`

```python
ensemble_distributions = self.ensemble_distributions(flat_x)
predicted_means = tf.stack(
    [unflatten(dist.mean()) for dist in ensemble_distributions], axis=-3
)
predicted_vars = tf.stack(
    [unflatten(dist.variance()) for dist in ensemble_distributions], axis=-3
)
```

**The Problem**:
- **Python list comprehension** iterates over Distribution objects
- Each `.mean()` and `.variance()` call is separate operation
- No vectorization across ensemble members
- Operations happen **outside compiled graph**

**TPU Impact**:
- **Sequential execution**: Each distribution processed one at a time
- **Multiple kernel launches**: Each method call = separate TPU kernel
- **CPU-TPU synchronization**: Python loop may execute on CPU
- **No batching**: Can't leverage TPU's parallel capabilities

**Why This Hurts**:
- TPU excels at **batched operations** across ensemble dimension
- Current: N separate operations
- TPU prefers: Single vectorized operation

**Recommendation**:
- If model outputs raw tensors, use **vectorized operations**:
  ```python
  # Model outputs: [batch, ensemble_size, 2] where last dim is [mean, variance]
  means = outputs[..., 0]
  variances = outputs[..., 1]
  # All operations are vectorized across ensemble dimension
  ```

### 5. **Memory Alignment and Padding Issues (MEDIUM)**

**Location**: TFP distribution internals

**The Problem**:
- TFP distributions use **broadcasting** for batch/event shapes
- TPUs require tensor dimensions to be **multiples of 8 or 128** for optimal performance
- TFP's broadcasting may result in **non-aligned shapes**
- XLA compiler adds **significant padding** to align memory

**TPU Impact**:
- **Memory waste**: Padding adds overhead
- **Inefficient operations**: Processing mostly zeros
- **v5e sensitivity**: v5e is particularly sensitive to memory alignment

**Example**:
- Per-core batch size 16 (not multiple of 8)
- Small output dimensions (common in BO)
- TFP broadcasting creates shapes that need padding
- TPU processes mostly padding, not actual data

**Recommendation**:
- Ensure batch sizes are **multiples of 8** (ideally 128)
- Use raw tensors with explicit shape control
- Avoid TFP's automatic broadcasting

### 6. **IndependentNormal vs MultivariateNormalTriL (MEDIUM)**

**Location**: `trieste/models/keras/architectures.py:323-338`

**The Problem**:
- `MultivariateNormalTriL` uses **triangular matrix** for covariance
- More complex than `IndependentNormal` (diagonal covariance)
- Additional operations for Cholesky decomposition
- More complex graph for XLA to compile

**TPU Impact**:
- **More complex graph**: Triangular matrix operations harder to optimize
- **More operations**: Additional matrix operations per forward pass
- **Less efficient**: IndependentNormal would be faster if correlations not needed

**Recommendation**:
- Use `IndependentNormal` if correlations not needed
- Or implement covariance computation manually with better TPU optimization

## Recommended Solutions

### Option 1: Replace TFP Layers with Raw Tensors (Best for TPU)

**Changes Required**:

1. **Modify GaussianNetwork output**:
   ```python
   # Instead of DistributionLambda, output raw tensors
   parameter_layer = tf_keras.layers.Dense(2, ...)(input_tensor)
   mean = parameter_layer[..., 0:1]
   log_var = parameter_layer[..., 1:2]  # or variance directly
   output = tf.concat([mean, log_var], axis=-1)
   ```

2. **Implement manual Gaussian NLL loss**:
   ```python
   def gaussian_nll(y_true, y_pred):
       mean, log_var = tf.split(y_pred, 2, axis=-1)
       var = tf.math.softplus(log_var) + 1e-6  # Ensure positive
       return tf.reduce_mean(
           0.5 * tf.math.log(2 * np.pi * var) + 
           0.5 * tf.square(y_true - mean) / var
       )
   ```

3. **Vectorize ensemble operations**:
   ```python
   # Model outputs: [batch, ensemble_size, 2]
   # All operations vectorized across ensemble dimension
   means = outputs[..., 0]
   variances = outputs[..., 1]
   ```

**Benefits**:
- **10-20x speedup** on TPU expected
- Simpler graph for XLA to optimize
- All operations in compiled graph
- Vectorized across ensemble members

### Option 2: Optimize TFP Usage (Partial Fix)

**Changes Required**:

1. **Use `convert_to_tensor_fn=None`** and extract statistics manually
2. **Ensure all operations are in `tf.function`**
3. **Use stateless random operations** for sampling
4. **Batch distribution operations** where possible

**Benefits**:
- Less code changes required
- Still uses TFP (may be needed for other reasons)
- Moderate improvement (2-3x expected)

## Impact Assessment

### Current Performance (with TFP)
- **TPU v2**: Significantly slower than P100 GPU
- **TPU v5e**: Much slower than v2 (architecture mismatch)
- **GPU**: Acceptable performance (handles TFP better)

### Expected Performance (with Raw Tensors)
- **TPU v2**: Should match or exceed P100 GPU
- **TPU v5e**: Should be competitive with v2
- **GPU**: Similar or slightly better performance

## Implementation Priority

1. **CRITICAL**: Replace DistributionLambda with raw tensors
2. **CRITICAL**: Implement manual Gaussian NLL loss
3. **HIGH**: Vectorize ensemble operations (remove Python loops)
4. **MEDIUM**: Optimize memory alignment (batch size multiples of 8)

## Testing Recommendations

1. **Profile with TPU profiler**:
   - Identify where TFP operations occur
   - Measure compilation time vs execution time
   - Check for CPU-TPU synchronization

2. **Compare TFP vs raw tensor implementation**:
   - Create proof-of-concept with raw tensors
   - Benchmark both on same TPU
   - Measure speedup

3. **Check XLA compilation**:
   - Use `tf.xla.experimental.compile` to inspect HLO graph
   - Compare graph complexity
   - Identify optimization opportunities

## Conclusion

The use of TensorFlow Probability `DistributionLambda` and `Distribution` objects in `GaussianNetwork` creates significant TPU performance bottlenecks that don't affect GPU performance as severely. The primary issues are:

1. **XLA compilation challenges** with lambda functions
2. **Python-level abstractions** breaking the compiled graph
3. **Non-vectorized operations** processing distributions sequentially
4. **Complex log-probability computation** creating inefficient kernels

Replacing TFP layers with raw tensor operations and manual loss computation should provide **10-20x speedup** on TPU while maintaining the same mathematical behavior.
