# Why TensorFlow Probability Works Fine on GPU but Struggles on TPU

## Executive Summary

TensorFlow Probability's abstraction layer works acceptably on GPU due to GPU's flexible execution model, lower compilation requirements, and better Python interoperability. TPU's strict XLA compilation, static graph requirements, and high kernel launch overhead make TFP abstractions significantly less efficient.

## Key Architectural Differences

### 1. **Execution Model: Flexible vs Static**

#### GPU Execution Model
- **Eager execution support**: Can execute operations immediately without full graph compilation
- **Dynamic graphs**: Can handle operations that vary in shape/size at runtime
- **Flexible kernel launches**: Can execute many small operations efficiently
- **Mixed execution**: Can seamlessly mix Python operations with GPU operations

#### TPU Execution Model
- **Static compilation required**: Everything must be compiled to XLA/HLO (High-Level Optimizer) graph
- **Fixed shapes**: Requires fixed tensor shapes at compile time (or recompilation)
- **Batch operations preferred**: Optimized for large batched operations, not many small ones
- **Graph breaks**: Python operations break the compiled graph, forcing CPU execution

**Impact on TFP**:
- **GPU**: TFP Distribution objects can be created and used with minimal compilation overhead
- **TPU**: TFP abstractions must be fully compiled, creating complex graphs that XLA struggles to optimize

### 2. **Compilation Requirements**

#### GPU Compilation
- **Optional compilation**: Can use `tf.function` for optimization, but not required
- **Gradual optimization**: Can optimize parts of graph incrementally
- **Less strict**: Can handle dynamic operations, variable shapes, Python callbacks
- **Faster compilation**: Less aggressive optimization = faster compile times

#### TPU Compilation
- **Mandatory compilation**: All operations must go through XLA compiler
- **Aggressive optimization**: XLA tries to fuse operations, optimize memory access
- **Very strict**: Requires static shapes, fixed control flow, no Python operations
- **Slow compilation**: Complex graphs take minutes to compile

**Impact on TFP**:
- **GPU**: TFP's lambda functions and Distribution objects compile reasonably well
- **TPU**: XLA struggles with TFP's complex abstractions, creating bloated graphs

### 3. **Kernel Launch Overhead**

#### GPU Kernel Launches
- **Low overhead**: ~1-10 microseconds per kernel launch
- **Many small kernels**: Can efficiently execute thousands of small operations
- **Parallel execution**: Can launch multiple kernels concurrently
- **Flexible scheduling**: GPU scheduler handles kernel queuing efficiently

#### TPU Kernel Launches
- **High overhead**: ~100-1000 microseconds per kernel launch
- **Few large kernels**: Prefers fewer, larger operations
- **Sequential execution**: Kernels execute more sequentially
- **Systolic array**: Optimized for large matrix operations, not many small ones

**Impact on TFP**:
- **GPU**: Multiple small TFP operations (`.mean()`, `.variance()`, `.log_prob()`) have acceptable overhead
- **TPU**: Each TFP operation = separate kernel launch with high overhead

### 4. **Memory Access Patterns**

#### GPU Memory Access
- **Flexible patterns**: Can handle various memory access patterns efficiently
- **Caching**: Good caching for irregular access patterns
- **Broadcasting**: Handles broadcasting operations efficiently
- **Dynamic shapes**: Can handle variable-sized tensors reasonably well

#### TPU Memory Access
- **Strict alignment**: Requires dimensions to be multiples of 8 or 128
- **Systolic array**: Optimized for regular, predictable access patterns
- **Broadcasting overhead**: Broadcasting may require padding/reshaping
- **Fixed shapes**: Variable shapes trigger recompilation

**Impact on TFP**:
- **GPU**: TFP's broadcasting and shape handling work efficiently
- **TPU**: TFP broadcasting may create non-aligned shapes requiring padding, wasting memory

### 5. **Python Interoperability**

#### GPU Python Interoperability
- **Seamless mixing**: Can mix Python operations with GPU operations
- **Eager execution**: Python code can execute alongside GPU operations
- **Callbacks**: Python callbacks (like in Keras) execute efficiently
- **Dynamic behavior**: Can handle dynamic Python logic

#### TPU Python Interoperability
- **Graph breaks**: Python operations break the compiled graph
- **CPU execution**: Python code executes on CPU, not TPU
- **Synchronization overhead**: CPU-TPU synchronization is expensive
- **Static requirements**: Requires static graph structure

**Impact on TFP**:
- **GPU**: Python loops over distributions (`[dist.mean() for dist in distributions]`) execute reasonably
- **TPU**: Python loops force CPU execution, breaking graph, causing expensive sync

### 6. **Lambda Functions and Dynamic Behavior**

#### GPU Lambda Handling
- **Reasonable compilation**: Lambda functions compile to GPU kernels reasonably well
- **Dynamic behavior**: Can handle some dynamic behavior in lambdas
- **Fallback options**: Can fall back to eager execution if compilation fails

#### TPU Lambda Handling
- **Strict compilation**: Lambda functions must be fully compiled to XLA
- **Static requirements**: XLA requires static control flow
- **Graph complexity**: Lambdas create complex graphs that XLA struggles to optimize
- **No fallback**: Must compile or fail

**Impact on TFP**:
- **GPU**: `DistributionLambda` with lambda functions compile and execute reasonably
- **TPU**: Lambda functions create complex, inefficient XLA graphs

## Specific TFP Operations: GPU vs TPU

### Distribution Object Creation

**GPU**:
```python
dist = tfp.distributions.Normal(mean, scale)
# Creates object efficiently, minimal overhead
```

**TPU**:
```python
dist = tfp.distributions.Normal(mean, scale)
# Must compile entire TFP Normal class logic to XLA
# Creates complex graph with validation, broadcasting, etc.
```

### Distribution Method Calls (`.mean()`, `.variance()`, `.log_prob()`)

**GPU**:
```python
mean = dist.mean()
# Low overhead kernel launch
# Can execute immediately
```

**TPU**:
```python
mean = dist.mean()
# High overhead kernel launch
# Must be in compiled graph
# If in Python loop, breaks graph → CPU execution
```

### DistributionLambda with Lambda Functions

**GPU**:
```python
distribution = tfp.layers.DistributionLambda(
    make_distribution_fn=lambda inputs: tfp.distributions.Normal(...)
)
# Compiles reasonably well
# Can handle dynamic behavior
```

**TPU**:
```python
distribution = tfp.layers.DistributionLambda(
    make_distribution_fn=lambda inputs: tfp.distributions.Normal(...)
)
# XLA struggles to optimize lambda
# Creates bloated graph
# May not fuse operations efficiently
```

### Python Loops Over Distributions

**GPU**:
```python
means = [dist.mean() for dist in distributions]
# Each .mean() launches GPU kernel with low overhead
# Reasonable performance
```

**TPU**:
```python
means = [dist.mean() for dist in distributions]
# Python loop executes on CPU
# Each .mean() requires CPU-TPU sync
# Very expensive
```

## Why GPU Handles TFP Better: Summary

1. **Flexible Execution**: GPU can handle dynamic operations and Python code better
2. **Lower Overhead**: Small kernel launches are efficient on GPU
3. **Better Compilation**: TFP abstractions compile reasonably well for GPU
4. **Python Interoperability**: Python operations don't break GPU execution as severely
5. **Memory Flexibility**: GPU handles TFP's broadcasting and shape handling efficiently

## Why TPU Struggles with TFP: Summary

1. **Strict Compilation**: Everything must compile to XLA, creating complex graphs
2. **High Overhead**: Many small operations = many expensive kernel launches
3. **Graph Breaks**: Python operations break compiled graph, forcing CPU execution
4. **Static Requirements**: TFP's dynamic behavior conflicts with TPU's static requirements
5. **Memory Alignment**: TFP broadcasting may create non-aligned shapes

## Performance Impact

### GPU Performance with TFP
- **Acceptable**: TFP overhead is manageable
- **Reasonable speed**: GPU handles TFP abstractions without major slowdown
- **Works as-is**: Current implementation performs acceptably

### TPU Performance with TFP
- **Significant slowdown**: TFP creates major bottlenecks
- **10-20x slower**: Compared to raw tensor implementation
- **Needs optimization**: Current implementation is not TPU-friendly

## Conclusion

GPU's flexible execution model, lower compilation requirements, and better Python interoperability make it well-suited for TensorFlow Probability's abstraction layer. TPU's strict XLA compilation, static graph requirements, and high kernel launch overhead make TFP abstractions significantly less efficient.

**Key Takeaway**: TFP works on GPU because GPU can "forgive" the abstraction overhead. TPU cannot, requiring everything to be optimized for its specific architecture.

This is why replacing TFP layers with raw tensors provides such dramatic speedup on TPU (10-20x) while having less impact on GPU (maybe 1.5-2x improvement).
