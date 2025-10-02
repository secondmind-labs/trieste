#!/usr/bin/env python3
"""
Enhanced GPU FESBO Benchmark - Comprehensive GPU performance validation
"""

import time
import tensorflow as tf
from statistics import mean, stdev
from trieste.space import Box, TaggedProductSearchSpace, DiscreteSearchSpace, CategoricalSearchSpace

class GPUFesboValidator:
    """GPU-specific FESBO performance validator."""
    
    def __init__(self, num_initial_samples_per_elt=1000):
        self._num_initial_samples_per_elt = num_initial_samples_per_elt
        self._initial_sampler_batch_per_elt = 1
        
    def validate_gpu_environment(self):
        """Validate GPU setup and provide detailed info."""
        
        print("🔍 GPU Environment Validation")
        print("=" * 50)
        
        # Check TensorFlow GPU support
        gpu_available = tf.test.is_gpu_available()
        gpu_built = tf.test.is_built_with_gpu_support()
        
        print(f"TensorFlow built with GPU: {gpu_built}")
        print(f"GPU available: {gpu_available}")
        
        # List physical devices
        physical_gpus = tf.config.list_physical_devices('GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        
        print(f"Physical GPUs: {len(physical_gpus)}")
        print(f"Logical GPUs: {len(logical_gpus)}")
        
        if physical_gpus:
            for i, gpu in enumerate(physical_gpus):
                print(f"  GPU {i}: {gpu}")
                
            # Check memory growth setting
            try:
                memory_growth = tf.config.experimental.get_memory_growth(physical_gpus[0])
                print(f"Memory growth enabled: {memory_growth}")
            except:
                print("Memory growth: Unable to determine")
                
        print()
        
        return gpu_available and len(physical_gpus) > 0
    
    def create_test_spaces(self):
        """Create different space types for comprehensive testing."""
        
        spaces = {
            'pure_continuous': TaggedProductSearchSpace([
                Box([0.0], [1.0]),
                Box([0.0], [10.0]),
                Box([-1.0], [1.0]),
                Box([0.0], [100.0])
            ]),
            
            'mixed_space': TaggedProductSearchSpace([
                Box([0.0], [1.0]),                    # Continuous
                Box([0.0], [10.0]),                   # Continuous
                DiscreteSearchSpace(tf.constant([[1], [2], [3], [4], [5]], dtype=tf.float64)),  # Discrete
                Box([0.0], [100.0])                   # Continuous
            ]),
            
            'pure_discrete': TaggedProductSearchSpace([
                DiscreteSearchSpace(tf.constant([[1], [2], [3], [4], [5]], dtype=tf.float64)),
                DiscreteSearchSpace(tf.constant([[10], [20], [30], [40]], dtype=tf.float64)),
                DiscreteSearchSpace(tf.constant([[100], [200], [300]], dtype=tf.float64))
            ])
        }
        
        return spaces
    
    def create_variable_subspace_spaces(self, num_subspaces):
        """Create search spaces with variable numbers of subspaces for scaling tests."""
        
        spaces = {}
        
        # Pure continuous space (all Box subspaces)
        box_subspaces = [Box([0.0], [1.0]) for _ in range(num_subspaces)]
        spaces['continuous'] = TaggedProductSearchSpace(box_subspaces)
        
        # Mixed space (alternating Box and Discrete)
        mixed_subspaces = []
        for i in range(num_subspaces):
            if i % 2 == 0:
                mixed_subspaces.append(Box([0.0], [1.0]))
            else:
                mixed_subspaces.append(DiscreteSearchSpace(tf.constant([[1], [2], [3]], dtype=tf.float64)))
        spaces['mixed'] = TaggedProductSearchSpace(mixed_subspaces)
        
        # Pure discrete space (all DiscreteSearchSpace subspaces)
        discrete_subspaces = [
            DiscreteSearchSpace(tf.constant([[1], [2], [3]], dtype=tf.float64)) 
            for _ in range(num_subspaces)
        ]
        spaces['discrete'] = TaggedProductSearchSpace(discrete_subspaces)
        
        return spaces
    
    def benchmark_fesbo_pattern(self, space, num_samples, device_name, num_iterations=100):
        """Benchmark the exact FESBO sampling pattern."""
        
        batch_size = num_samples * self._initial_sampler_batch_per_elt
        
        print(f"  🎯 FESBO Pattern: {num_iterations} iterations of sample({batch_size})")
        
        # Test sequential sampling
        seq_times = []
        with tf.device(device_name):
            for _ in range(3):  # 3 test runs
                start = time.perf_counter()
                
                for _ in range(num_iterations):
                    sample = space.sample(batch_size)
                    # Force GPU computation
                    _ = tf.reduce_sum(sample).numpy()
                
                seq_times.append(time.perf_counter() - start)
        
        # Test parallel sampling  
        par_times = []
        with tf.device(device_name):
            for _ in range(3):  # 3 test runs
                start = time.perf_counter()
                
                for _ in range(num_iterations):
                    sample = space.sample_parallel(batch_size)
                    # Force GPU computation
                    _ = tf.reduce_sum(sample).numpy()
                
                par_times.append(time.perf_counter() - start)
        
        seq_mean = mean(seq_times)
        par_mean = mean(par_times)
        improvement = (seq_mean - par_mean) / seq_mean * 100
        
        return {
            'sequential': seq_mean,
            'parallel': par_mean,
            'improvement_pct': improvement,
            'seq_std': stdev(seq_times) if len(seq_times) > 1 else 0,
            'par_std': stdev(par_times) if len(par_times) > 1 else 0
        }
    
    def run_subspace_scaling_benchmark(self, device_name, sample_sizes=[1000, 2000, 5000]):
        """Run subspace scaling benchmark with large sample sizes."""
        
        print(f"\n{'='*25} SUBSPACE SCALING BENCHMARK {'='*25}")
        print(f"Device: {device_name}")
        print("Testing how parallel sampling scales with number of subspaces")
        print("(FESBO supports up to 20-dimensional search spaces)")
        print()
        
        # Test different numbers of subspaces
        subspace_counts = [2, 4, 6, 8, 10, 12, 16, 20]
        
        all_results = {}
        
        for num_samples in sample_sizes:
            print(f"\n{'='*15} BATCH SIZE: {num_samples:,} SAMPLES {'='*15}")
            
            results_for_size = {}
            
            for num_subspaces in subspace_counts:
                print(f"\n🔍 {num_subspaces} Subspaces:")
                print("-" * 30)
                
                try:
                    spaces = self.create_variable_subspace_spaces(num_subspaces)
                    results_for_subspaces = {}
                    
                    for space_type, space in spaces.items():
                        print(f"\n  📈 {space_type.title()} ({num_subspaces}D → {space.dimension}D total):")
                        
                        try:
                            # Fewer iterations for large samples to avoid memory issues
                            iterations = 10 if num_samples >= 5000 else 15 if num_samples >= 2000 else 20
                            
                            results = self.benchmark_fesbo_pattern(
                                space, num_samples, device_name, num_iterations=iterations
                            )
                            
                            # Calculate samples per second for context
                            seq_rate = num_samples * iterations / results['sequential'] * 3  # 3 runs
                            par_rate = num_samples * iterations / results['parallel'] * 3
                            
                            print(f"    Sequential: {results['sequential']:.4f}s ({seq_rate:,.0f} samples/sec)")
                            print(f"    Parallel:   {results['parallel']:.4f}s ({par_rate:,.0f} samples/sec)")
                            print(f"    Improvement: {results['improvement_pct']:+.1f}%")
                            
                            if results['improvement_pct'] > 10:
                                print("    🎉 SIGNIFICANT WIN! Parallel dominates!")
                            elif results['improvement_pct'] > 5:
                                print("    ✅ Good improvement")
                            elif results['improvement_pct'] > 0:
                                print("    ✓ Modest improvement")
                            else:
                                print("    ⚠️ Sequential still better")
                            
                            results_for_subspaces[space_type] = results
                            
                        except Exception as e:
                            print(f"    ❌ Error: {str(e)[:50]}...")
                            if 'memory' in str(e).lower() or 'oom' in str(e).lower():
                                print("    (Likely GPU memory issue - try smaller batch sizes)")
                            results_for_subspaces[space_type] = None
                    
                    results_for_size[num_subspaces] = results_for_subspaces
                    
                except Exception as e:
                    print(f"  ❌ Failed to create {num_subspaces} subspaces: {e}")
                    results_for_size[num_subspaces] = None
            
            all_results[num_samples] = results_for_size
        
        # Analyze scaling results
        self.analyze_subspace_scaling_results(all_results, device_name)
        
        return all_results
    
    def analyze_subspace_scaling_results(self, all_results, device_name):
        """Analyze subspace scaling results and provide insights."""
        
        print(f"\n{'='*60}")
        print(f"🧠 SUBSPACE SCALING ANALYSIS ({device_name})")
        print("="*60)
        
        break_even_found = {}
        best_improvements = {}
        
        for num_samples, size_results in all_results.items():
            print(f"\n📈 BATCH SIZE {num_samples:,} - Scaling Results:")
            print("-" * 50)
            
            for space_type in ['continuous', 'mixed', 'discrete']:
                print(f"\n{space_type.title()} Space:")
                improvements = []
                subspace_counts = []
                
                for num_subspaces, subspace_results in size_results.items():
                    if subspace_results and subspace_results.get(space_type):
                        improvement = subspace_results[space_type]['improvement_pct']
                        improvements.append(improvement)
                        subspace_counts.append(num_subspaces)
                        
                        status = "🎉 PARALLEL WINS" if improvement > 10 else "✅ GOOD" if improvement > 5 else "✓ MODEST" if improvement > 0 else "⚠️ SEQUENTIAL"
                        print(f"  {num_subspaces:2d} subspaces: {improvement:+6.1f}% ({status})")
                
                if improvements:
                    best_improvement = max(improvements)
                    best_subspaces = subspace_counts[improvements.index(best_improvement)]
                    
                    # Track best improvements
                    key = f"{space_type}_{num_samples}"
                    best_improvements[key] = (best_improvement, best_subspaces, num_samples)
                    
                    # Find break-even point
                    positive_improvements = [(imp, sub) for imp, sub in zip(improvements, subspace_counts) if imp > 0]
                    if positive_improvements:
                        break_even_point = min(positive_improvements, key=lambda x: x[1])[1]
                        break_even_found[key] = break_even_point
                        print(f"  🎯 Break-even: {break_even_point} subspaces")
                    else:
                        print(f"  📊 No break-even (best: {best_improvement:+.1f}% at {best_subspaces})")
        
        # Summary insights
        print(f"\n🎊 KEY INSIGHTS FOR {device_name}:")
        print("-" * 40)
        
        if break_even_found:
            min_break_even = min(break_even_found.values())
            print(f"✅ BREAK-EVEN FOUND!")
            print(f"• Minimum break-even: {min_break_even} subspaces")
            
            # Group by space type
            for space_type in ['continuous', 'mixed', 'discrete']:
                type_break_evens = [v for k, v in break_even_found.items() if k.startswith(space_type)]
                if type_break_evens:
                    min_be = min(type_break_evens)
                    print(f"• {space_type.title()}: {min_be}+ subspaces for parallel benefit")
        else:
            print("⚠️ No break-even points found")
            
        # Best performance summary
        overall_best = max(best_improvements.values(), key=lambda x: x[0]) if best_improvements else None
        if overall_best and overall_best[0] > 0:
            improvement, subspaces, samples = overall_best
            space_type = [k for k, v in best_improvements.items() if v == overall_best][0].split('_')[0]
            print(f"🏆 Best result: {improvement:+.1f}% ({space_type}, {subspaces} subspaces, {samples:,} samples)")
        
        print(f"\n💡 DEPLOYMENT RECOMMENDATIONS:")
        if '/GPU:' in device_name and break_even_found:
            min_break_even = min(break_even_found.values())
            print(f"• Enable parallel sampling for {min_break_even}+ subspaces on GPU")
            print(f"• Large batches (2000+) show best scaling")
            print(f"• Mixed/discrete spaces benefit most from parallelization")
        elif break_even_found:
            print(f"• Parallel sampling shows benefits on this device!")
            print(f"• Consider using for {min(break_even_found.values())}+ subspaces")
        else:
            print(f"• Sequential sampling remains optimal for this device")
            print(f"• Consider testing with even larger batch sizes")
            if '/CPU:' in device_name:
                print(f"• GPU deployment would likely show different results")
    
    def run_comprehensive_gpu_benchmark(self):
        """Run comprehensive FESBO benchmark (GPU when available, CPU otherwise)."""
        
        gpu_available = self.validate_gpu_environment()
        
        if gpu_available:
            print("🚀 Comprehensive GPU FESBO Benchmark")
            devices = ['/CPU:0', '/GPU:0']
        else:
            print("🚀 Comprehensive CPU FESBO Benchmark")
            print("⚠️  GPU not available - testing CPU performance only")
            devices = ['/CPU:0']
        
        print("=" * 50)
        
        # Run both original FESBO tests and subspace scaling tests
        for device in devices:
            print(f"\n{'='*20} {device} RESULTS {'='*20}")
            
            # 1. Original FESBO pattern tests
            print(f"\n🎯 ORIGINAL FESBO PATTERNS")
            print("-" * 40)
            
            spaces = self.create_test_spaces()
            test_configs = [
                (500, "Typical FESBO batch (lower)"),
                (750, "Typical FESBO batch (mid)"),
                (1000, "Typical FESBO batch (upper)")
            ]
            
            for space_name, space in spaces.items():
                print(f"\n📊 {space_name.upper().replace('_', ' ')} SPACE")
                print("-" * 30)
                
                for num_samples, description in test_configs:
                    print(f"\n{description} (num_samples={num_samples}):")
                    
                    try:
                        results = self.benchmark_fesbo_pattern(
                            space, num_samples, device, num_iterations=20
                        )
                        
                        print(f"    Sequential: {results['sequential']:.4f}s (±{results['seq_std']:.4f})")
                        print(f"    Parallel:   {results['parallel']:.4f}s (±{results['par_std']:.4f})")
                        print(f"    Improvement: {results['improvement_pct']:+.1f}%")
                        
                        if results['improvement_pct'] > 5:
                            print("    ✅ Significant improvement!")
                        elif results['improvement_pct'] > 0:
                            print("    ✓ Modest improvement") 
                        else:
                            print("    ⚠️ No improvement (sequential better)")
                            
                    except Exception as e:
                        print(f"    ❌ Error: {str(e)[:60]}...")
                        if '/GPU:' in device:
                            print("    (This is expected if GPU is not available)")
                        else:
                            print("    (Unexpected CPU error - check implementation)")
            
            # 2. Subspace scaling tests
            try:
                # Adjust sample sizes based on device capabilities
                if '/GPU:' in device:
                    sample_sizes = [1000, 2000, 5000]  # Larger sizes for GPU
                else:
                    sample_sizes = [1000, 2000]  # Smaller sizes for CPU to avoid long runs
                
                self.run_subspace_scaling_benchmark(device, sample_sizes)
                
            except Exception as e:
                print(f"\n❌ Subspace scaling benchmark failed on {device}: {e}")
                if 'memory' in str(e).lower():
                    print("This might indicate memory constraints - try smaller batch sizes.")

def provide_gpu_testing_guidance():
    """Provide guidance for comprehensive GPU testing."""
    
    print("\n" + "="*60)
    print("🎯 COMPREHENSIVE FESBO TESTING GUIDANCE")
    print("="*60)
    
    print("\n📋 WHAT THIS BENCHMARK TESTS:")
    print("• Original FESBO patterns (500-1000 samples, fixed subspaces)")
    print("• Subspace scaling (1000-5000 samples, 2-20 subspaces)")
    print("• Both CPU and GPU performance (when available)")
    print("• Mixed, continuous, and discrete search spaces")
    
    print("\n📈 WHAT TO LOOK FOR:")
    print("• Break-even points where parallel becomes faster")
    print("• GPU vs CPU performance differences")
    print("• Scaling trends with more subspaces")
    print("• Mixed/discrete spaces typically show better improvements")
    
    print("\n✅ SUCCESS INDICATORS:")
    print("• Positive improvements (>0%) indicate parallel benefits")
    print("• Break-even points found for 8-15 subspaces")
    print("• GPU shows better scaling than CPU")
    print("• Large batches (2000+) favor parallel sampling")
    print("• Mixed spaces show strongest scaling trends")
    
    print("\n⚠️  POTENTIAL ISSUES:")
    print("• GPU memory errors with very large batches")
    print("• First GPU runs slower due to compilation")
    print("• CPU may not show break-even even at large scales")
    print("• TensorFlow bridge overhead significant on CPU")
    
    print("\n🔧 IF RESULTS ARE POOR:")
    print("• Check GPU utilization with nvidia-smi during runs")
    print("• Verify GPU memory usage (should be <80% for stability)")
    print("• Reduce sample sizes if getting OOM errors")
    print("• Ensure TensorFlow detects GPU properly")
    print("• CPU results being slower is expected and normal")

def main():
    """Main GPU benchmark execution."""
    
    # Set TensorFlow to show device placement (optional)
    # tf.debugging.set_log_device_placement(True)
    
    print("🎯 FESBO GPU Performance Validator")
    print("=" * 60)
    print("This will test your exact FESBO sampling patterns on GPU")
    print("and validate the optimization benefits.\n")
    
    validator = GPUFesboValidator(num_initial_samples_per_elt=1000)
    
    try:
        validator.run_comprehensive_gpu_benchmark()
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        print("This might indicate GPU memory issues or configuration problems.")
    
    provide_gpu_testing_guidance()
    
    print(f"\n🎊 Next Steps:")
    print("• If break-even points found → implement smart thresholds in FESBO")
    print("• If GPU shows strong improvements → prioritize GPU deployment")
    print("• If mixed/discrete spaces improve → enable parallel for those types")
    print("• Use the specific subspace counts identified as break-even points")

if __name__ == "__main__":
    main()
