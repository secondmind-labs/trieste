#!/usr/bin/env python3
"""
Simple GPU Test - Minimal test for FESBO optimization validation
"""

import time
import tensorflow as tf
from trieste.space import Box, TaggedProductSearchSpace

def simple_gpu_test():
    """Simple, robust GPU test for FESBO optimization."""
    
    print("🔍 Simple GPU Test for FESBO Optimization")
    print("=" * 50)
    
    # Check GPU availability
    gpu_available = tf.test.is_gpu_available()
    physical_gpus = tf.config.list_physical_devices('GPU')
    
    print(f"GPU Available: {gpu_available}")
    print(f"Physical GPUs: {len(physical_gpus)}")
    
    if not gpu_available or len(physical_gpus) == 0:
        print("❌ No GPU detected - testing on CPU only")
        devices = ['/CPU:0']
    else:
        print("✅ GPU detected - testing CPU vs GPU")
        devices = ['/CPU:0', '/GPU:0']
    
    print()
    
    # Create simple test space (avoiding DiscreteSearchSpace issues)
    space = TaggedProductSearchSpace([
        Box([0.0], [1.0]),
        Box([0.0], [1.0]),
        Box([0.0], [1.0]),
        Box([0.0], [1.0]),
        Box([0.0], [1.0])  # 5 simple Box subspaces
    ])
    
    print("📊 Testing TaggedProductSearchSpace with 5 Box subspaces")
    print("This represents a common FESBO scenario.\n")
    
    # Test configurations
    test_configs = [
        (1, 10),    # 1 sample, 10 iterations
        (10, 10),   # 10 samples, 10 iterations  
        (100, 5),   # 100 samples, 5 iterations
    ]
    
    for device in devices:
        print(f"--- {device} Results ---")
        
        for num_samples, num_iterations in test_configs:
            print(f"\nTest: {num_samples} samples × {num_iterations} iterations")
            
            try:
                # Test sequential
                with tf.device(device):
                    start = time.perf_counter()
                    for _ in range(num_iterations):
                        sample = space.sample(num_samples)
                        _ = tf.reduce_sum(sample).numpy()  # Force execution
                    seq_time = time.perf_counter() - start
                
                # Test parallel
                with tf.device(device):
                    start = time.perf_counter()
                    for _ in range(num_iterations):
                        sample = space.sample_parallel(num_samples)
                        _ = tf.reduce_sum(sample).numpy()  # Force execution
                    par_time = time.perf_counter() - start
                
                improvement = (seq_time - par_time) / seq_time * 100
                
                print(f"  Sequential: {seq_time:.4f}s")
                print(f"  Parallel:   {par_time:.4f}s")
                print(f"  Improvement: {improvement:+.1f}%")
                
                if improvement > 5:
                    print("  ✅ Good improvement!")
                elif improvement > 0:
                    print("  ✓ Modest improvement")
                else:
                    print("  ⚠️ Sequential faster")
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:50]}...")
        
        print()
    
    print("🎯 Interpretation:")
    print("• For pure Box spaces (continuous): expect modest/no improvement")
    print("• This is normal - Box spaces use optimized sampling already")
    print("• Mixed spaces (Box + Discrete) would show better improvement")
    print("• GPU should show better scaling than CPU for larger batches")


def check_smart_threshold_behavior():
    """Test the smart threshold logic."""
    
    print("\n🧠 Smart Threshold Behavior Test")
    print("=" * 40)
    
    space = TaggedProductSearchSpace([Box([0.0], [1.0])] * 10)  # 10 subspaces
    
    # Test different workload sizes
    test_cases = [
        (1, "tiny workload"),
        (5, "small workload"), 
        (50, "medium workload"),
        (500, "large workload")
    ]
    
    gpu_available = tf.test.is_gpu_available()
    
    print(f"GPU Available: {gpu_available}")
    print("Testing which cases trigger parallel sampling:\n")
    
    for num_samples, description in test_cases:
        workload = num_samples * 10  # 10 subspaces
        
        # Simulate the threshold logic
        if gpu_available:
            min_workload = 50
            min_subspaces = 3
        else:
            min_workload = 1000
            min_subspaces = 10
            
        would_use_parallel = (workload >= min_workload or 
                             (10 >= min_subspaces and num_samples >= 5))
        
        print(f"{description:15s}: {num_samples:3d} samples, workload={workload:4d}, "
              f"parallel={'YES' if would_use_parallel else 'NO '}")


if __name__ == "__main__":
    # Reduce TensorFlow verbosity
    tf.get_logger().setLevel('ERROR')
    
    simple_gpu_test()
    check_smart_threshold_behavior()
    
    print("\n🎊 Next Steps:")
    print("• If you see any improvement → the optimization works!")
    print("• Even modest improvement (5-10%) is worthwhile") 
    print("• GPU should perform better than CPU for larger workloads")
    print("• Ready to implement the one-line change in fesbo.py!")
