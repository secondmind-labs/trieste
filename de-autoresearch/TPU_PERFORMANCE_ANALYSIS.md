# DeepEnsemble TPU Performance Analysis

## Executive Summary

The DeepEnsemble model shows significantly slower training on TPU (v2 and v5e) compared to P100 GPU due to several architectural and data pipeline issues that prevent efficient TPU utilization. **Even with batch size 128 and TPUStrategy**, the model suffers from per-core batch size issues, dictionary-based data structures, non-vectorized ensemble architecture, CPU-bound preprocessing, and **critical TensorFlow Probability (TFP) performance issues** in the output layers.

## Critical Performance Issues

### 1. **Ineffective Per-Core Batch Size (CRITICAL)**

**Location**: Training configuration with TPUStrategy
- Global batch size: **128**
- Per-core batch size: **16** (128 ÷ 8 TPU cores)
- TPU Impact: **SEVERE**

**The Problem**: 
- Cloud TPUs (v2-8, v5e) have **8 cores**
- `TPUStrategy` automatically **shards** the global batch across cores
- Global batch size 128 → **16 samples per core**
- TPU MXUs are optimized for 128x128 or 8x128 vector registers
- Batch size 16 is far too small to saturate MXU registers
- Hardware sits idle waiting for data

**Why This Matters**:
- Each TPU core processes only 16 samples at a time
- MXU underutilization: registers are mostly empty
- High kernel launch overhead relative to computation
- Poor memory bandwidth utilization
- TPU v5e is even more sensitive due to its LLM-optimized architecture

**Recommendation**: 
- Increase **global batch size to 1024+** (128+ per core)
- For v5e, consider 2048+ (256+ per core) if memory allows
- This is the **single most important fix**

### 2. **Dictionary-Based Multi-Input/Output Architecture (CRITICAL)**

**Location**: `trieste/models/keras/models.py:163-168, 208-235`

**The Problem**:
- Model compiled with: `loss=[loss_fn] * ensemble_size` and `metrics=[metrics] * ensemble_size`
- Creates **N separate input layers** and **N separate output layers** (one per ensemble member)
- Data fed as **dictionaries** mapping names to tensors: `{input_name: tensor, ...}`
- Each ensemble member is a **separate branch** in the computation graph

**TPU Impact**:
- **No vectorization**: Ensemble members processed independently, not as batched operations
- **Complex graph compilation**: TPU/XLA must compile N separate branches
- **Dictionary overhead**: `tf.data.Dataset.from_tensor_slices((dict, dict))` is inefficient on TPU
- **Multiple small operations**: Instead of one large matrix multiply, N small ones
- **Graph dispatch overhead**: Each branch requires separate TPU kernel dispatch

**Why TPU Struggles**:
- TPUs excel at **large batched matrix multiplications**
- Current architecture: N small parallel branches
- TPU prefers: Single vectorized operation across ensemble dimension
- Dictionary lookups add overhead in compiled graph

**Recommendation**: 
- **Vectorize ensemble**: Restructure to single model with ensemble dimension
- Use stacked tensors: `[batch, ensemble_size, features]` instead of dictionaries
- Single forward pass processes all ensemble members simultaneously
- This is a **major architectural change** but critical for TPU performance

### 3. **CPU-Bound Data Preparation and Bootstrap Sampling**

**Location**: `trieste/models/keras/models.py:208-235` (`prepare_dataset`), `trieste/models/keras/utils.py:64-88`

**The Problem**:
- `prepare_dataset()` runs on **CPU** (host)
- Creates Python dictionaries: `inputs = {}`, `outputs = {}`
- If bootstrap enabled: `sample_with_replacement()` runs on CPU for each ensemble member
- Uses `tf.random.uniform()` and `tf.gather()` which may execute on CPU
- Data then transferred from CPU to TPU

**TPU Impact**:
- **High host-to-device latency**: TPU waits for CPU preprocessing
- **Many small transfers**: Dictionary entries transferred separately
- **Bootstrap overhead**: CPU-bound random sampling for each ensemble member
- **Pipeline stalls**: TPU sits idle while CPU prepares next batch
- TPU has **higher CPU-TPU transfer latency** than GPU

**Recommendation**: 
- Move data preparation **into TPU graph** using `tf.function`
- Generate bootstrap indices on TPU: `tf.random.stateless_uniform()` inside `@tf.function`
- Pre-compute all bootstrap indices in single TPU operation
- Use dense tensors instead of dictionaries

### 4. **Inefficient tf.data.Dataset Construction with Dictionaries**

**Location**: `trieste/models/keras/models.py:514-526` (`_build_tf_dataset`)

```python
tf_dataset = tf.data.Dataset.from_tensor_slices((x, y))  # x, y are dictionaries
```

**Issues**:
- `from_tensor_slices()` with **dictionaries** is particularly slow on TPU
- TPU prefers dense tensor slices, not dictionary mappings
- Prefetch only enabled conditionally (when `steps_per_epoch` is set)
- Missing TPU-specific data optimizations

**TPU Impact**:
- Dictionary unpacking overhead in data pipeline
- Suboptimal data transfer patterns
- TPU data pipeline not optimized for dictionary structures

**Recommendation**:
- Restructure to use **tuple of tensors** instead of dictionaries
- Enable `experimental_optimization` and `experimental_deterministic=False`
- Always enable prefetching: `.prefetch(tf.data.AUTOTUNE)`
- Consider TFRecord format for very large datasets

### 5. **Graph Compilation and XLA Overhead**

**Location**: Model architecture and compilation

**The Problem**:
- TPU uses **XLA (Accelerated Linear Algebra)** to compile computation graphs
- Deep Ensemble creates **complex graph** with N separate branches
- Each ensemble member = separate subgraph
- Dictionary-based inputs/outputs create complex control flow

**TPU Impact**:
- **Long compilation time**: XLA must optimize N separate branches
- **Inefficient compilation**: XLA struggles with dictionary-based multi-input models
- **Recompilation triggers**: If shapes vary (bootstrap, padding), graph recompiles
- **Fixed shape requirements**: TPU requires fixed shapes for efficient compilation
- **Dispatch overhead**: Multiple kernel dispatches instead of single batched operation

**Why This Hurts**:
- First training step includes compilation time (can be minutes)
- If shapes change, recompilation causes "hanging" behavior
- Complex graphs compile less efficiently than simple vectorized operations

**Recommendation**:
- Vectorize ensemble to **single unified graph**
- Ensure **fixed tensor shapes** throughout
- Use static shapes where possible (avoid dynamic padding)
- Profile compilation time separately from training time

### 6. **TPU v5e Architecture Mismatch**

**Location**: Hardware-specific

**The Problem**:
- TPU v5e is **optimized for LLMs and Transformers**
- Architecture tuned for large attention mechanisms and matrix operations
- **Not optimized** for small fully-connected layers (typical of Deep Ensembles)
- Different memory bandwidth characteristics than v2/v3

**TPU Impact**:
- v5e shows **worse performance** than v2 for this workload
- Memory bandwidth limits hit before compute limits
- Architectural overhead for small-scale operations
- Mega-core configuration not beneficial for ensemble models

**Why v5e is Slower**:
- Optimized for different workload patterns
- Small FC layers don't utilize v5e's strengths
- Memory bandwidth becomes bottleneck
- Overhead of managing specialized compute units

**Recommendation**:
- Consider **TPU v2/v3** instead of v5e for Deep Ensembles
- Or increase model size/complexity to better utilize v5e
- Profile memory vs compute utilization

### 7. **Early Stopping and Callback Overhead**

**Location**: `trieste/models/keras/models.py:150-153`

Early stopping callback:
- Monitors loss every epoch
- Requires **CPU-TPU synchronization** to check stopping condition
- May restore best weights (additional transfer)

**TPU Impact**:
- Frequent synchronization between TPU and CPU is expensive
- **TPU-CPU communication latency is higher** than GPU-CPU
- Each epoch check involves data transfer
- Can cause pipeline stalls

**Recommendation**:
- Increase patience or reduce monitoring frequency
- Consider TPU-native monitoring solutions
- Batch multiple epochs before checking
- Use `tf.summary` for logging instead of frequent callbacks

### 8. **TensorFlow Probability Performance Issues (CRITICAL)**

**Location**: `trieste/models/keras/architectures.py:340-353, 323-338`, `trieste/models/keras/models.py:385-391`

**The Problem**:
- `GaussianNetwork` uses `tfp.layers.DistributionLambda` with **Python lambda functions**
- Model outputs are **TFP Distribution objects**, not raw tensors
- Loss function calls `dist.log_prob()` which includes complex TFP logic
- Prediction extracts statistics using **Python loops**: `[dist.mean() for dist in distributions]`

**TPU Impact**:
- **XLA compilation issues**: Lambda functions don't compile efficiently to HLO graphs
- **Graph breaks**: Distribution method calls happen outside compiled graph
- **CPU-TPU synchronization**: Python loops may execute on CPU
- **Complex log_prob**: TFP's log_prob creates inefficient kernels with many small operations
- **No vectorization**: Distributions processed sequentially, not batched
- **Memory alignment**: TFP broadcasting may create non-aligned shapes requiring padding

**Why TPU Struggles**:
- TPU requires **static, well-optimized graphs** for efficient execution
- TFP Distribution objects are high-level Python abstractions
- Lambda functions introduce dynamic behavior XLA can't fully optimize
- GPU handles this better due to more flexible execution model

**Recommendation**:
- **Replace DistributionLambda with raw tensors**: Output `[mean, log_variance]` instead of Distribution
- **Implement manual Gaussian NLL**: Use pure TensorFlow operations for loss
- **Vectorize ensemble operations**: Remove Python loops, use `tf.stack` and `tf.reduce_mean`
- **Expected improvement**: 10-20x speedup on TPU

**See**: `TFP_TPU_ISSUES.md` for detailed analysis and solutions.

### 9. **Loss Computation Overhead**

**Location**: `trieste/models/keras/models.py:165`

```python
loss=[self.optimizer.loss] * model.ensemble_size
```

**The Problem**:
- Keras computes **N separate losses** (one per ensemble member)
- Losses are then **summed** for backpropagation
- Each loss computation is separate operation in graph
- Combined with TFP issues, this creates even more overhead

**TPU Impact**:
- Multiple small loss computations instead of single batched operation
- Additional graph complexity
- Less efficient than vectorized loss computation

**Recommendation**:
- If vectorizing ensemble, compute single vectorized loss
- Use `tf.reduce_mean` or `tf.reduce_sum` across ensemble dimension
- Ensure loss computation is part of main computation graph
- Replace TFP log_prob with manual Gaussian NLL (see TFP issues above)

### 9. **Validation Data Handling**

**Location**: `trieste/models/keras/models.py:484-490`

Validation data is created using:
```python
fit_args["validation_data"] = tf.data.Dataset.from_tensor_slices((x_val, y_val))
```

**TPU Impact**:
- Same issues as training data pipeline
- Additional CPU-TPU transfer for validation
- Validation happens every epoch (frequent transfers)

**Recommendation**:
- Optimize validation data pipeline similarly to training
- Consider validating less frequently
- Use TPU-optimized validation data construction

### 10. **Learning Rate Reset Overhead**

**Location**: `trieste/models/keras/models.py:507-510`

After training, learning rate is reset:
```python
self.optimizer.optimizer.lr.assign(self.original_lr)
```

**TPU Impact**:
- Requires TPU-CPU synchronization to read/write optimizer state
- Additional overhead if done frequently

**Recommendation**:
- Minimize optimizer state access
- Use learning rate schedules instead of manual reset

## TPU-Specific Architecture Differences

### TPU v2 vs v5e
- **TPU v5e**: More sensitive to small batches due to different MXU architecture
- **TPU v5e**: Higher memory bandwidth but requires larger batches to saturate
- Both suffer from the same issues, but v5e shows worse performance due to architectural changes

### TPU vs GPU Differences
- **TPU**: Higher latency for small operations (favors large batches)
- **TPU**: More efficient at large matrix multiplications
- **TPU**: Higher CPU-TPU transfer latency than GPU
- **TPU**: Prefers static shapes and compiled functions

## Recommended Fixes (Priority Order)

### Critical Priority (Required for TPU Performance)
1. **Increase global batch size to 1024+** (128+ per core)
   - This is the **single most important fix**
   - Expected: 5-10x speedup
   - For v5e: Consider 2048+ (256+ per core)

2. **Vectorize ensemble architecture**
   - Restructure to single model with ensemble dimension
   - Replace dictionary inputs with stacked tensors: `[batch, ensemble_size, features]`
   - Single forward pass processes all ensemble members
   - Expected: 3-5x speedup

### High Priority (Major Impact)
3. **Move data preparation to TPU**
   - Use `tf.function` for `prepare_dataset`
   - Generate bootstrap indices on TPU
   - Use dense tensors instead of dictionaries
   - Expected: 2-3x speedup

4. **Optimize data pipeline**
   - Replace dictionary-based `from_tensor_slices` with tuple of tensors
   - Enable `experimental_optimization`
   - Always enable prefetching
   - Expected: 1.5-2x speedup

### Medium Priority (Significant Impact)
5. **Reduce callback overhead**
   - Increase early stopping patience
   - Reduce monitoring frequency
   - Use TPU-native logging
   - Expected: 10-20% improvement

6. **Consider TPU v2/v3 instead of v5e**
   - v5e optimized for LLMs, not small FC ensembles
   - v2/v3 may perform better for this workload
   - Profile to compare

### Low Priority (Incremental Improvements)
7. **Optimize validation data pipeline**
8. **Minimize optimizer state access**
9. **Ensure fixed tensor shapes** (avoid recompilation)

## Expected Performance Improvements

With **critical fixes** (batch size + vectorization):
- **10-20x speedup** potential
- Should match or exceed P100 GPU performance

With **all optimizations**:
- **20-50x speedup** potential
- May exceed GPU performance significantly

## Implementation Strategy

### Phase 1: Quick Wins (No Architecture Changes)
1. Increase global batch size to 1024+
2. Optimize data pipeline (remove dictionaries from Dataset)
3. Move bootstrap to TPU graph

### Phase 2: Architecture Changes (Larger Effort)
1. Vectorize ensemble architecture
2. Replace dictionary inputs with stacked tensors
3. Single unified computation graph

### Phase 3: Fine-tuning
1. Profile with TPU profiler
2. Optimize based on profiling results
3. Consider TPU v2/v3 if v5e still underperforms

## Testing Recommendations

1. **Profile with TPU profiler** to identify bottlenecks
   - Check if "Input Bound" or "Compute Bound"
   - Measure compilation time vs training time
   - Identify CPU-TPU transfer overhead

2. **Compare batch sizes**: 128, 256, 512, 1024, 2048
   - Measure per-core batch size impact
   - Find optimal for your model size

3. **Measure data pipeline overhead** separately
   - Profile `prepare_dataset` time
   - Measure dictionary vs tensor performance
   - Test bootstrap overhead

4. **Test with and without bootstrap**
   - Bootstrap adds significant overhead
   - Consider if necessary for your use case

5. **Compare TPU v2 vs v5e**
   - v5e may not be optimal for this workload
   - Profile both if available

6. **Profile CPU-TPU transfer times**
   - Identify data transfer bottlenecks
   - Optimize transfer patterns

7. **Test vectorized vs dictionary-based architecture**
   - Create proof-of-concept vectorized version
   - Compare performance directly
