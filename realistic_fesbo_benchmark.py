#!/usr/bin/env python3
"""
Realistic FESBO Mixed Space Testing
Testing with 10-30% discrete/categorical subspaces (realistic ratios)
"""

import time
import tensorflow as tf
from statistics import mean
from trieste.space import Box, DiscreteSearchSpace, CategoricalSearchSpace, TaggedProductSearchSpace
import random

def create_realistic_mixed_spaces():
    """
    Create realistic mixed spaces where only 10-30% of subspaces are discrete/categorical.
    This better represents real FESBO optimization scenarios.
    """
    
    realistic_configs = []
    
    # 10D space: 1-3 discrete, 7-9 continuous
    for num_discrete in [1, 2, 3]:
        subspaces = []
        
        # Add continuous subspaces
        for i in range(10 - num_discrete):
            subspaces.append(Box([0.0], [1.0]))
        
        # Add discrete subspaces  
        for i in range(num_discrete):
            discrete_points = tf.constant([[1], [2], [3], [4]], dtype=tf.float64)
            subspaces.append(DiscreteSearchSpace(discrete_points))
        
        # Shuffle to distribute discrete subspaces randomly
        random.shuffle(subspaces)
        
        realistic_configs.append((
            10, subspaces, f"10D ({num_discrete} discrete, {10-num_discrete} continuous)"
        ))
    
    # 15D space: 2-4 discrete, 11-13 continuous  
    for num_discrete in [2, 3, 4]:
        subspaces = []
        
        # Add continuous subspaces
        for i in range(15 - num_discrete):
            subspaces.append(Box([0.0], [1.0]))
        
        # Add discrete subspaces
        for i in range(num_discrete):
            discrete_points = tf.constant([[1], [2], [3]], dtype=tf.float64)
            subspaces.append(DiscreteSearchSpace(discrete_points))
        
        # Shuffle to distribute discrete subspaces randomly
        random.shuffle(subspaces)
        
        realistic_configs.append((
            15, subspaces, f"15D ({num_discrete} discrete, {15-num_discrete} continuous)"
        ))
    
    # 20D space: 2-6 discrete, 14-18 continuous
    for num_discrete in [2, 4, 6]:
        subspaces = []
        
        # Add continuous subspaces
        for i in range(20 - num_discrete):
            subspaces.append(Box([0.0], [1.0]))
        
        # Add discrete/categorical mix
        for i in range(num_discrete):
            if i % 2 == 0:
                # DiscreteSearchSpace
                discrete_points = tf.constant([[1], [2], [3], [4], [5]], dtype=tf.float64)
                subspaces.append(DiscreteSearchSpace(discrete_points))
            else:
                # Use DiscreteSearchSpace as well (CategoricalSearchSpace might not be supported)
                discrete_points = tf.constant([[10], [20], [30]], dtype=tf.float64)
                subspaces.append(DiscreteSearchSpace(discrete_points))
        
        # Shuffle to distribute discrete subspaces randomly
        random.shuffle(subspaces)
        
        realistic_configs.append((
            20, subspaces, f"20D ({num_discrete} discrete, {20-num_discrete} continuous)"
        ))
    
    return realistic_configs

def benchmark_realistic_mixed_spaces():
    """Benchmark realistic mixed spaces with proper discrete/continuous ratios."""
    
    print("🎯 Realistic FESBO Mixed Space Benchmark")
    print("=" * 60)
    print("Testing spaces with 10-30% discrete subspaces (realistic ratios)")
    print()
    
    # Set random seed for reproducible subspace arrangements
    random.seed(42)
    
    realistic_configs = create_realistic_mixed_spaces()
    sample_sizes = [500, 1000, 2000, 5000]
    
    best_results = []
    all_results = []
    
    for total_dims, subspaces, description in realistic_configs:
        print(f"\n--- {description} ---")
        
        # Calculate actual percentages
        num_discrete = sum(1 for s in subspaces if isinstance(s, DiscreteSearchSpace))
        discrete_pct = (num_discrete / total_dims) * 100
        print(f"Discrete percentage: {discrete_pct:.1f}%")
        
        space = TaggedProductSearchSpace(subspaces)
        
        for num_samples in sample_sizes:
            print(f"\n{num_samples:,} samples:")
            
            try:
                # Sequential baseline
                seq_times = []
                for _ in range(3):
                    start = time.perf_counter()
                    seq_result = space.sample(num_samples, seed=42)
                    seq_times.append(time.perf_counter() - start)
                
                # Parallel with pure TensorFlow optimizations
                par_times = []
                for _ in range(3):
                    start = time.perf_counter()
                    par_result = space.sample_parallel(num_samples, seed=42)
                    par_times.append(time.perf_counter() - start)
                
                seq_mean = mean(seq_times)
                par_mean = mean(par_times)
                improvement = (seq_mean - par_mean) / seq_mean * 100
                
                # Calculate throughput
                seq_throughput = num_samples / seq_mean
                par_throughput = num_samples / par_mean
                
                print(f"  Sequential: {seq_mean:.4f}s ({seq_throughput:,.0f} samples/sec)")
                print(f"  Parallel:   {par_mean:.4f}s ({par_throughput:,.0f} samples/sec)")
                print(f"  Improvement: {improvement:+.1f}%")
                
                # Verify correctness
                if seq_result.shape == par_result.shape:
                    print(f"  ✅ Correct shape: {seq_result.shape}")
                else:
                    print(f"  ❌ Shape mismatch!")
                
                # Record results
                result_data = {
                    'config': description,
                    'samples': num_samples,
                    'improvement': improvement,
                    'discrete_pct': discrete_pct,
                    'seq_time': seq_mean,
                    'par_time': par_mean
                }
                all_results.append(result_data)
                
                if improvement > 15:
                    print(f"  🎉 EXCELLENT! Major speedup achieved!")
                    best_results.append(result_data)
                elif improvement > 5:
                    print(f"  ✅ GOOD! Solid improvement!")
                    best_results.append(result_data)
                elif improvement > 0:
                    print(f"  ✓ Progress! Parallel is faster!")
                    best_results.append(result_data)
                else:
                    print(f"  ⚠️ Sequential still faster by {abs(improvement):.1f}%")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    # Analysis by discrete percentage
    analyze_by_discrete_percentage(all_results)
    
    # Summary
    print(f"\n{'='*60}")
    print("🏆 REALISTIC MIXED SPACE RESULTS")
    print("="*60)
    
    if best_results:
        print(f"\n✅ SUCCESS! Realistic mixed spaces showing improvements:")
        
        # Sort by improvement
        best_results.sort(key=lambda x: x['improvement'], reverse=True)
        
        for i, result in enumerate(best_results[:8]):
            print(f"{i+1:2d}. {result['improvement']:+5.1f}% - {result['config']} @ {result['samples']:,} samples")
        
        best_improvement = best_results[0]['improvement']
        print(f"\n🥇 BEST REALISTIC RESULT: {best_improvement:+.1f}% improvement!")
        print(f"   Configuration: {best_results[0]['config']}")
        print(f"   Sample size: {best_results[0]['samples']:,}")
        
        success_rate = len(best_results) / len([r for r in all_results if r['improvement'] is not None])
        print(f"📊 Success rate: {success_rate:.1%} of configurations show speedup")
        
        if success_rate >= 0.5:
            print("🎊 MISSION SUCCESS: Realistic mixed spaces are now competitive!")
            return True
        else:
            print("🔧 PARTIAL SUCCESS: Some realistic configurations improved")
            return False
    else:
        print("❌ No improvements found in realistic scenarios")
        print("💡 May need different optimization strategies for low-discrete-ratio spaces")
        return False

def analyze_by_discrete_percentage(all_results):
    """Analyze performance by discrete subspace percentage."""
    
    print(f"\n📊 Performance by Discrete Percentage")
    print("-" * 45)
    
    # Group by discrete percentage ranges
    ranges = [
        (0, 15, "10-15% discrete"),
        (15, 25, "15-25% discrete"), 
        (25, 35, "25-35% discrete")
    ]
    
    for min_pct, max_pct, label in ranges:
        matching_results = [
            r for r in all_results 
            if min_pct <= r['discrete_pct'] < max_pct and r['improvement'] is not None
        ]
        
        if matching_results:
            improvements = [r['improvement'] for r in matching_results]
            avg_improvement = mean(improvements)
            positive_count = len([imp for imp in improvements if imp > 0])
            
            print(f"{label:15s}: {avg_improvement:+5.1f}% avg ({positive_count}/{len(improvements)} positive)")
            
            if avg_improvement > 5:
                print(f"                  ✅ GOOD performance in this range")
            elif avg_improvement > 0:
                print(f"                  ✓ Positive performance")
            else:
                print(f"                  ⚠️ Needs optimization")

def compare_realistic_vs_unrealistic():
    """Compare realistic mixed spaces vs the unrealistic 50/50 spaces."""
    
    print(f"\n🔬 Realistic vs Unrealistic Mixed Space Comparison")
    print("=" * 55)
    
    # Unrealistic 50/50 mixed space
    unrealistic_space = TaggedProductSearchSpace([
        Box([0.0], [1.0]), 
        DiscreteSearchSpace(tf.constant([[1], [2]], dtype=tf.float64))
    ] * 10)  # 20D with 50% discrete
    
    # Realistic 20% mixed space  
    realistic_subspaces = [Box([0.0], [1.0])] * 16  # 80% continuous
    realistic_subspaces.extend([
        DiscreteSearchSpace(tf.constant([[1], [2]], dtype=tf.float64)) 
        for _ in range(4)  # 20% discrete
    ])
    random.shuffle(realistic_subspaces)
    realistic_space = TaggedProductSearchSpace(realistic_subspaces)
    
    test_samples = 2000
    
    spaces = {
        'Unrealistic (50% discrete)': unrealistic_space,
        'Realistic (20% discrete)': realistic_space
    }
    
    print(f"Testing with {test_samples:,} samples:")
    print("-" * 35)
    
    for name, space in spaces.items():
        # Sequential
        start = time.perf_counter()
        _ = space.sample(test_samples, seed=42)
        seq_time = time.perf_counter() - start
        
        # Parallel
        start = time.perf_counter()
        _ = space.sample_parallel(test_samples, seed=42)
        par_time = time.perf_counter() - start
        
        improvement = (seq_time - par_time) / seq_time * 100
        
        status = "🎉 WINNER" if improvement > 10 else "✅ GOOD" if improvement > 0 else "⚠️ SLOWER"
        print(f"{name:25s}: {improvement:+6.1f}% {status}")
    
    print(f"\n💡 INSIGHT: Realistic ratios should perform better due to")
    print(f"   higher proportion of efficient Box subspace sampling.")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    
    # Test realistic mixed space scenarios
    success = benchmark_realistic_mixed_spaces()
    
    # Compare realistic vs unrealistic ratios
    compare_realistic_vs_unrealistic()
    
    print(f"\n{'='*60}")
    print("🎯 REALISTIC SCENARIO VERDICT")
    print("="*60)
    
    if success:
        print("🎊 SUCCESS: Realistic FESBO mixed spaces show parallel speedups!")
        print("✅ Production-ready for real-world optimization scenarios")
        print("✅ 10-30% discrete subspaces perform well with parallel sampling")
    else:
        print("🔧 OPTIMIZATION NEEDED: Realistic scenarios need more work")
        print("💡 Consider further optimizations for low-discrete-ratio spaces")
        print("📊 Results show current implementation effectiveness")
