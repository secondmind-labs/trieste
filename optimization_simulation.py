#!/usr/bin/env python3
"""
Simulation script for optimization use case.
Tests the realistic performance of sampling in batch optimization loops.
"""

import time
import tensorflow as tf
from statistics import mean, stdev
from trieste.space import Box, TaggedProductSearchSpace

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

    @property
    def elapsed_s(self):
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time

    @classmethod
    def time(cls):
        return cls()


class OptimizationSimulator:
    """Simulates the optimization sampling pattern from your use case."""
    
    def __init__(self, initial_sampler_batch_per_elt=10, num_initial_samples_per_elt=20):
        self._initial_sampler_batch_per_elt = initial_sampler_batch_per_elt
        self._num_initial_samples_per_elt = num_initial_samples_per_elt
        self._seed = 42
        
    def simulate_initial_point_sampler(self, space, num_samples, batch_idx=0, use_parallel=False):
        """Simulate the initial_point_sampler from your optimization code."""
        
        batch_size = int(num_samples * self._initial_sampler_batch_per_elt)
        num_initial_samples = int(self._num_initial_samples_per_elt * num_samples)
        
        samples = []
        total_sample_calls = 0
        
        # Simulate the sampling loop
        for i in range(0, num_initial_samples, batch_size):
            if use_parallel and hasattr(space, 'sample_parallel'):
                sample = space.sample_parallel(
                    batch_size, 
                    seed=self._seed + batch_idx * num_initial_samples + i
                )
            else:
                sample = space.sample(
                    batch_size,
                    seed=self._seed + batch_idx * num_initial_samples + i
                )
            
            # Reshape as in your code
            reshaped = tf.reshape(
                sample,
                [self._initial_sampler_batch_per_elt, num_samples, -1]
            )
            samples.append(reshaped)
            total_sample_calls += 1
            
        return samples, total_sample_calls


def benchmark_optimization_simulation():
    """Benchmark the optimization simulation with realistic parameters."""
    
    print("=== Optimization Use Case Simulation ===")
    print("Testing realistic batch optimization sampling patterns\n")
    
    # Create test spaces
    box_space = Box([0.0] * 10, [1.0] * 10)
    product_space = TaggedProductSearchSpace([Box([0.0], [1.0])] * 10)
    
    # Realistic optimization parameters
    test_configs = [
        {"num_samples": 1, "description": "Single sample optimization"},
        {"num_samples": 5, "description": "Small batch optimization"}, 
        {"num_samples": 10, "description": "Medium batch optimization"},
        {"num_samples": 20, "description": "Large batch optimization"},
    ]
    
    simulator = OptimizationSimulator(
        initial_sampler_batch_per_elt=10,  # Realistic batch multiplier
        num_initial_samples_per_elt=50      # Realistic initial samples
    )
    
    num_repeats = 3
    
    for config in test_configs:
        num_samples = config["num_samples"]
        description = config["description"]
        
        print(f"\n--- {description} (num_samples={num_samples}) ---")
        
        # Test Box space (baseline)
        box_times = []
        for _ in range(num_repeats):
            with Timer.time() as timer:
                samples, calls = simulator.simulate_initial_point_sampler(
                    box_space, num_samples, use_parallel=False
                )
            box_times.append(timer.elapsed_s)
        
        # Test TaggedProductSearchSpace (sequential)
        sequential_times = []
        for _ in range(num_repeats):
            with Timer.time() as timer:
                samples, calls = simulator.simulate_initial_point_sampler(
                    product_space, num_samples, use_parallel=False
                )
            sequential_times.append(timer.elapsed_s)
            
        # Test TaggedProductSearchSpace (parallel)
        parallel_times = []
        for _ in range(num_repeats):
            with Timer.time() as timer:
                samples, calls = simulator.simulate_initial_point_sampler(
                    product_space, num_samples, use_parallel=True
                )
            parallel_times.append(timer.elapsed_s)
        
        # Calculate stats
        box_mean = mean(box_times)
        box_std = stdev(box_times) if len(box_times) > 1 else 0
        
        seq_mean = mean(sequential_times)
        seq_std = stdev(sequential_times) if len(sequential_times) > 1 else 0
        
        par_mean = mean(parallel_times)
        par_std = stdev(parallel_times) if len(parallel_times) > 1 else 0
        
        print(f"Sample calls per simulation: {calls}")
        print(f"Box (baseline):        {box_mean:.4f}s (±{box_std:.4f})")
        print(f"Product (sequential):  {seq_mean:.4f}s (±{seq_std:.4f})")  
        print(f"Product (parallel):    {par_mean:.4f}s (±{par_std:.4f})")
        
        # Calculate improvements
        seq_vs_box = (seq_mean / box_mean - 1) * 100
        par_vs_seq = (par_mean / seq_mean - 1) * 100
        par_vs_box = (par_mean / box_mean - 1) * 100
        
        print(f"Sequential vs Box:     {seq_vs_box:+.1f}% slower")
        print(f"Parallel vs Sequential: {par_vs_seq:+.1f}%")
        print(f"Parallel vs Box:       {par_vs_box:+.1f}% slower")


def test_repeated_sampling():
    """Test the specific case of repeated small sampling calls."""
    
    print("\n\n=== Repeated Small Sampling Test ===")
    print("Testing repeated calls with small batch sizes (realistic for optimization)\n")
    
    product_space = TaggedProductSearchSpace([Box([0.0], [1.0])] * 10)
    
    # Test repeated small samples (like in optimization loop)
    batch_sizes = [10, 50, 100]
    num_iterations = 20
    
    for batch_size in batch_sizes:
        print(f"--- Batch size: {batch_size}, Iterations: {num_iterations} ---")
        
        # Sequential approach
        with Timer.time() as seq_timer:
            for i in range(num_iterations):
                _ = product_space.sample(batch_size, seed=42 + i)
        
        # Parallel approach  
        with Timer.time() as par_timer:
            for i in range(num_iterations):
                _ = product_space.sample_parallel(batch_size, seed=42 + i)
        
        print(f"Sequential: {seq_timer.elapsed_s:.4f}s")
        print(f"Parallel:   {par_timer.elapsed_s:.4f}s")
        print(f"Improvement: {(seq_timer.elapsed_s / par_timer.elapsed_s - 1) * 100:+.1f}%\n")


if __name__ == "__main__":
    # Set TensorFlow to be less verbose
    tf.get_logger().setLevel('ERROR')
    
    benchmark_optimization_simulation()
    test_repeated_sampling()
    
    print("\n=== Summary ===")
    print("This simulation shows performance in realistic optimization scenarios")
    print("where sample() is called repeatedly with smaller batch sizes.")
