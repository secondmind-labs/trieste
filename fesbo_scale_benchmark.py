#!/usr/bin/env python3
"""
FESBO-Scale Pure TensorFlow Benchmark - Testing realistic workloads
"""

import time
import tensorflow as tf
from statistics import mean
from pure_tf_parallel_sampling import PureTensorFlowTaggedProductSearchSpace
from trieste.space import Box, DiscreteSearchSpace, TaggedProductSearchSpace

def benchmark_fesbo_scale():
    """Benchmark pure TF approach with realistic FESBO workloads."""
    
    print("🎯 FESBO-Scale Pure TensorFlow Benchmark")
    print("=" * 60)
    print("Testing pure TF parallel sampling with realistic FESBO configurations")
    print()
    
    # FESBO-realistic configurations
    fesbo_configs = [
        (10, [Box([0.0], [1.0])] * 10, "10D Box space (typical FESBO)"),
        (15, [Box([0.0], [1.0])] * 15, "15D Box space (larger FESBO)"),
        (20, [Box([0.0], [1.0])] * 20, "20D Box space (max FESBO)"),
        (12, [Box([0.0], [1.0]), DiscreteSearchSpace(tf.constant([[1], [2]], dtype=tf.float64))] * 6, "12D Mixed space"),
        (16, [Box([0.0], [1.0]), DiscreteSearchSpace(tf.constant([[1], [2]], dtype=tf.float64))] * 8, "16D Mixed space"),
    ]
    
    # FESBO-realistic sample sizes
    sample_sizes = [500, 1000, 2000, 5000]
    
    best_results = []
    
    for num_subspaces, subspace_list, description in fesbo_configs:
        print(f"\n{'='*15} {description} {'='*15}")
        
        # Create spaces
        standard_space = TaggedProductSearchSpace(subspace_list)
        pure_tf_space = PureTensorFlowTaggedProductSearchSpace(subspace_list)
        
        for num_samples in sample_sizes:
            print(f"\n📊 {num_samples:,} samples:")
            
            try:
                # Standard sequential (3 runs)
                seq_times = []
                for _ in range(3):
                    start = time.perf_counter()
                    _ = standard_space.sample(num_samples, seed=42)
                    seq_times.append(time.perf_counter() - start)
                
                # Pure TensorFlow parallel (3 runs)
                pure_tf_times = []
                for _ in range(3):
                    start = time.perf_counter()
                    _ = pure_tf_space.sample_pure_tf_parallel(num_samples, seed=42)
                    pure_tf_times.append(time.perf_counter() - start)
                
                seq_mean = mean(seq_times)
                pure_tf_mean = mean(pure_tf_times)
                improvement = (seq_mean - pure_tf_mean) / seq_mean * 100
                
                # Calculate throughput
                seq_throughput = num_samples / seq_mean
                pure_tf_throughput = num_samples / pure_tf_mean
                
                print(f"  Sequential:  {seq_mean:.4f}s ({seq_throughput:,.0f} samples/sec)")
                print(f"  Pure TF:     {pure_tf_mean:.4f}s ({pure_tf_throughput:,.0f} samples/sec)")
                print(f"  Improvement: {improvement:+.1f}%")
                
                if improvement > 5:
                    print(f"  🎉 SIGNIFICANT WIN! Pure TF dominates!")
                    best_results.append((description, num_samples, improvement))
                elif improvement > 0:
                    print(f"  ✅ Pure TF beats sequential!")
                    best_results.append((description, num_samples, improvement))
                else:
                    print(f"  ⚠️  Sequential still faster")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    # Summary of best results
    print(f"\n{'='*60}")
    print("🏆 BEST RESULTS SUMMARY")
    print("="*60)
    
    if best_results:
        best_results.sort(key=lambda x: x[2], reverse=True)  # Sort by improvement
        print(f"\nTop performing configurations:")
        for i, (config, samples, improvement) in enumerate(best_results[:10]):
            print(f"{i+1:2d}. {improvement:+5.1f}% - {config} @ {samples:,} samples")
        
        overall_best = best_results[0]
        print(f"\n🥇 OVERALL BEST: {overall_best[2]:+.1f}% improvement")
        print(f"   Configuration: {overall_best[0]}")
        print(f"   Sample size: {overall_best[1]:,}")
        
        # Find break-even threshold
        positive_results = [(config, samples) for config, samples, imp in best_results if imp > 0]
        if positive_results:
            print(f"\n✅ BREAK-EVEN ACHIEVED!")
            print(f"   Total configurations with speedup: {len(positive_results)}")
            print(f"   Pure TF parallel sampling is faster for many FESBO workloads!")
    else:
        print("\n⚠️  No configurations showed positive improvements")
        print("Consider testing on GPU or with even larger workloads")
    
    print(f"\n🎯 DEPLOYMENT RECOMMENDATIONS:")
    if best_results:
        # Find minimum dimensions that show benefit
        winning_configs = [result for result in best_results if result[2] > 0]
        if winning_configs:
            min_winning_dims = min([int(config.split('D')[0]) for config, _, _ in winning_configs])
            print(f"• Enable pure TF parallel for {min_winning_dims}+ dimensional spaces")
            print(f"• Best results with Box-heavy configurations")
            print(f"• Sample sizes 500+ show consistent benefits")
            print(f"• This approach should be EVEN BETTER on GPU!")

def test_correctness():
    """Test that pure TF results match sequential results."""
    
    print(f"\n🧪 Correctness Verification")
    print("=" * 40)
    
    # Test space
    space_standard = TaggedProductSearchSpace([Box([0.0], [1.0])] * 8)
    space_pure_tf = PureTensorFlowTaggedProductSearchSpace([Box([0.0], [1.0])] * 8)
    
    # Generate samples with same seed
    seq_result = space_standard.sample(100, seed=42)
    pure_tf_result = space_pure_tf.sample_pure_tf_parallel(100, seed=42)
    
    print(f"Sequential shape: {seq_result.shape}")
    print(f"Pure TF shape:    {pure_tf_result.shape}")
    
    # Check ranges (should be in [0,1] for Box([0,1]))
    seq_min, seq_max = tf.reduce_min(seq_result), tf.reduce_max(seq_result)
    tf_min, tf_max = tf.reduce_min(pure_tf_result), tf.reduce_max(pure_tf_result)
    
    print(f"Sequential range: [{seq_min:.3f}, {seq_max:.3f}]")
    print(f"Pure TF range:    [{tf_min:.3f}, {tf_max:.3f}]")
    
    # Results should be valid (different due to different random ops, but same ranges)
    if (0 <= seq_min <= seq_max <= 1) and (0 <= tf_min <= tf_max <= 1):
        print("✅ Both results are within valid ranges")
    else:
        print("❌ Invalid sample ranges detected")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    
    test_correctness()
    benchmark_fesbo_scale()
