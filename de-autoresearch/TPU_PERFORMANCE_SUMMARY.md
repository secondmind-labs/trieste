# DeepEnsemble TPU Performance Issues - Summary

## Context
Experiments using **batch size 128** and **TPUStrategy** still show significantly slower training on TPU (v2 and v5e) compared to P100 GPU.

## Root Causes (Ranked by Impact)

### 🔴 CRITICAL: Per-Core Batch Size Too Small
**Problem**: Global batch size 128 ÷ 8 TPU cores = **16 samples per core**

- TPU MXUs need **128+ samples per core** to saturate registers
- Batch size 16 leaves hardware mostly idle
- Each core processes tiny batches inefficiently

**Fix**: Increase **global batch size to 1024+** (128+ per core)
- For TPU v5e: Consider 2048+ (256+ per core)
- **Expected improvement: 5-10x speedup**

### 🔴 CRITICAL: Non-Vectorized Ensemble Architecture
**Problem**: Ensemble uses **dictionary-based multi-input/output** structure

- Model has N separate input/output layers (one per ensemble member)
- Data fed as dictionaries: `{input_name: tensor, ...}`
- Each ensemble member = separate branch in computation graph
- No vectorization across ensemble members

**Why TPU Struggles**:
- TPUs excel at **large batched matrix multiplications**
- Current: N small parallel branches
- TPU prefers: Single vectorized operation

**Fix**: **Vectorize ensemble architecture**
- Restructure to single model with ensemble dimension
- Use stacked tensors: `[batch, ensemble_size, features]`
- Single forward pass processes all members simultaneously
- **Expected improvement: 3-5x speedup**

### 🟠 HIGH: Dictionary-Based Data Pipeline
**Problem**: `tf.data.Dataset.from_tensor_slices((dict, dict))` is inefficient on TPU

- Dictionary unpacking overhead
- TPU prefers dense tensor operations
- Current: Dictionary lookups in data pipeline

**Fix**: Replace dictionaries with **tuple of tensors**
- Use `(stacked_inputs, stacked_outputs)` instead of dictionaries
- Enable TPU-specific optimizations
- **Expected improvement: 1.5-2x speedup**

### 🟠 HIGH: CPU-Bound Data Preparation
**Problem**: `prepare_dataset()` runs on CPU, then transfers to TPU

- Bootstrap sampling (if enabled) happens on CPU
- Python dictionary construction on CPU
- High host-to-device transfer latency

**Fix**: Move data preparation **into TPU graph**
- Use `tf.function` for `prepare_dataset`
- Generate bootstrap indices on TPU
- **Expected improvement: 2-3x speedup**

### 🔴 CRITICAL: TensorFlow Probability Performance Issues
**Problem**: TFP `DistributionLambda` and Distribution objects create TPU bottlenecks

- `DistributionLambda` with lambda functions don't compile well with XLA
- Model outputs Distribution objects instead of raw tensors
- Python loops extract statistics: `[dist.mean() for dist in distributions]`
- TFP's `log_prob` creates complex, inefficient kernels
- Operations happen outside compiled graph

**Why TPU Struggles**:
- TPU requires static, optimized graphs
- TFP abstractions break graph compilation
- GPU handles this better due to flexible execution

**Fix**: **Replace TFP layers with raw tensors**
- Output `[mean, log_variance]` instead of Distribution objects
- Implement manual Gaussian NLL loss
- Vectorize ensemble operations (remove Python loops)
- **Expected improvement: 10-20x speedup**

**See**: `TFP_TPU_ISSUES.md` for detailed analysis

### 🟡 MEDIUM: TPU v5e Architecture Mismatch
**Problem**: TPU v5e optimized for LLMs/Transformers, not small FC ensembles

- v5e architecture tuned for different workload patterns
- Memory bandwidth limits hit before compute limits
- Shows worse performance than v2 for this workload

**Fix**: Consider **TPU v2/v3** instead of v5e, or increase model complexity

### 🟡 MEDIUM: Graph Compilation Overhead
**Problem**: Complex graph with N separate branches compiles inefficiently

- XLA must optimize N separate subgraphs
- Dictionary-based inputs create complex control flow
- Recompilation if shapes vary

**Fix**: Vectorized architecture creates simpler, more efficient graph

### 🟢 LOW: Callback Overhead
**Problem**: Early stopping causes frequent CPU-TPU synchronization

**Fix**: Increase patience, reduce monitoring frequency

## Recommended Action Plan

### Phase 1: Quick Wins (Immediate)
1. ✅ **Increase global batch size to 1024+**
   - Single line change in training config
   - Biggest impact, minimal code changes

2. ✅ **Optimize data pipeline**
   - Replace dictionary `from_tensor_slices` with tuple of tensors
   - Enable TPU optimizations

3. ✅ **Move bootstrap to TPU**
   - Generate indices in `tf.function` on TPU

### Phase 2: Architecture Changes (Larger Effort)
1. **Vectorize ensemble architecture**
   - Major refactoring but critical for TPU
   - Replace dictionary inputs with stacked tensors
   - Single unified computation graph

## Expected Combined Impact

With **Phase 1 fixes** (batch size + data pipeline):
- **5-10x speedup** expected
- Should significantly improve TPU performance

With **Phase 1 + Phase 2** (full vectorization):
- **10-20x speedup** expected
- Should match or exceed P100 GPU performance

## Key Insight

The fundamental issue is that **TPUs are optimized for large batched operations**, but the current DeepEnsemble architecture:
1. Processes ensemble members **independently** (not batched)
2. Uses **small per-core batches** (16 instead of 128+)
3. Relies on **dictionary-based structures** (not dense tensors)

The solution requires both **increasing batch size** and **vectorizing the ensemble architecture** to fully utilize TPU capabilities.
