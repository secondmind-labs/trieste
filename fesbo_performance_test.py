#!/usr/bin/env python3
"""
Test FESBO-like sampling patterns to understand GPU vs CPU performance.
"""

import time
import tensorflow as tf
from statistics import mean, stdev
from trieste.space import Box, TaggedProductSearchSpace

class FesboSamplingSimulator:
    """Simulate the exact FESBO sampling pattern."""
    
    def __init__(self, num_initial_samples_per_elt=1000, initial_sampler_batch_per_elt=1):
        self._num_initial_samples_per_elt = num_initial_samples_per_elt
        self._initial_sampler_batch_per_elt = initial_sampler_batch_per_elt
        
    def simulate_fesbo_sampling(self, space, num_samples, use_parallel=False):
        """Simulate exact FESBO sampling pattern."""
        batch_size = int(num_samples * self._initial_sampler_batch_per_elt)
        num_initial_samples = int(self._num_initial_samples_per_elt * num_samples)
        
        samples = []
        sample_calls = 0
        
        # This is the exact loop from FESBO
        for i in range(0, num_initial_samples, batch_size):
            if use_parallel and hasattr(space, 'sample_parallel'):
                sample = space.sample_parallel(batch_size)
            else:
                sample = space.sample(batch_size)
            
            samples.append(tf.reshape(
                sample,
                [self._initial_sampler_batch_per_elt, num_samples, -1],
            ))
            sample_calls += 1
            
        return samples, sample_calls

def test_device_performance():
    """Test performance on different devices."""
    
    print("=== FESBO Sampling Performance Test ===")
    print("Testing realistic FESBO sampling patterns\n")
    
    # Check available devices
    devices = []
    if tf.config.list_physical_devices('CPU'):
        devices.append('/CPU:0')
    if tf.config.list_physical_devices('GPU'):
        devices.append('/GPU:0')
        
    print(f"Available devices: {devices}\n")
    
    # Test configurations - realistic FESBO parameters
    test_configs = [
        {"num_samples": 1, "description": "Single sample (typical)"},
        {"num_samples": 10, "description": "Small batch"},
        {"num_samples": 100, "description": "Large batch"},
    ]
    
    # Reduced scale for testing - use 100 instead of 1000 iterations
    simulator = FesboSamplingSimulator(num_initial_samples_per_elt=100)
    
    # Create test spaces
    product_space = TaggedProductSearchSpace([Box([0.0], [1.0])] * 10)
    
    for device in devices:
        print(f"--- Testing on {device} ---")
        
        with tf.device(device):
            for config in test_configs:
                num_samples = config["num_samples"]
                description = config["description"]
                
                print(f"\n{description} (num_samples={num_samples}):")
                
                # Test sequential
                times_seq = []
                for _ in range(3):  # 3 repeats
                    start = time.perf_counter()
                    samples, calls = simulator.simulate_fesbo_sampling(
                        product_space, num_samples, use_parallel=False
                    )
                    # Force execution by accessing the data
                    _ = tf.stack(samples)
                    times_seq.append(time.perf_counter() - start)
                
                # Test parallel
                times_par = []
                for _ in range(3):  # 3 repeats
                    start = time.perf_counter()
                    samples, calls = simulator.simulate_fesbo_sampling(
                        product_space, num_samples, use_parallel=True
                    )
                    # Force execution by accessing the data
                    _ = tf.stack(samples)
                    times_par.append(time.perf_counter() - start)
                
                seq_mean = mean(times_seq)
                par_mean = mean(times_par)
                improvement = (seq_mean / par_mean - 1) * 100
                
                print(f"  Sample calls: {calls}")
                print(f"  Sequential: {seq_mean:.4f}s (±{stdev(times_seq):.4f})")
                print(f"  Parallel:   {par_mean:.4f}s (±{stdev(times_par):.4f})")
                print(f"  Improvement: {improvement:+.1f}%")
        
        print()

def test_smart_threshold():
    """Test if the smart threshold is appropriate for FESBO use case."""
    
    print("=== Smart Threshold Analysis ===")
    
    # FESBO typical parameters
    simulator = FesboSamplingSimulator(num_initial_samples_per_elt=1000)
    product_space = TaggedProductSearchSpace([Box([0.0], [1.0])] * 10)
    
    test_samples = [1, 5, 10, 20, 50, 100]
    
    for num_samples in test_samples:
        num_subspaces = len(product_space.subspace_tags)
        workload_size = num_samples * num_subspaces
        
        batch_size = num_samples  # FESBO batch size
        num_calls = 1000  # FESBO iterations
        total_workload = workload_size * num_calls
        
        # Current threshold logic
        would_use_parallel = workload_size >= 10000
        
        print(f"num_samples={num_samples:3d}: "
              f"workload={workload_size:5d}, "
              f"calls={num_calls}, "
              f"total={total_workload:8d}, "
              f"parallel={would_use_parallel}")

def main():
    # Set TensorFlow to be less verbose
    tf.get_logger().setLevel('ERROR')
    
    # Run tests
    test_device_performance()
    test_smart_threshold()
    
    print("\n=== FESBO Recommendations ===")
    print("1. Test on your actual GPU setup to measure real performance")
    print("2. Consider the total workload (1000 iterations * workload_size)")
    print("3. The smart threshold may need GPU-specific tuning")
    print("4. Profile memory usage - GPU memory constraints may matter")

if __name__ == "__main__":
    main()
