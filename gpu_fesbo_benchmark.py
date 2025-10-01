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
                DiscreteSearchSpace(tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)),  # Discrete
                Box([0.0], [100.0])                   # Continuous
            ]),
            
            'pure_discrete': TaggedProductSearchSpace([
                DiscreteSearchSpace(tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)),
                DiscreteSearchSpace(tf.constant([10, 20, 30, 40], dtype=tf.float64)),
                DiscreteSearchSpace(tf.constant([100, 200, 300], dtype=tf.float64))
            ])
        }
        
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
    
    def run_comprehensive_gpu_benchmark(self):
        """Run comprehensive GPU benchmark."""
        
        if not self.validate_gpu_environment():
            print("❌ GPU not available or not properly configured!")
            print("   Make sure you have:")
            print("   • GPU drivers installed")
            print("   • TensorFlow-GPU installed")
            print("   • CUDA/cuDNN properly configured")
            return
        
        print("🚀 Comprehensive GPU FESBO Benchmark")
        print("=" * 50)
        
        spaces = self.create_test_spaces()
        test_configs = [
            (1, "Single sample (most common in FESBO)"),
            (10, "Small batch"),
            (50, "Medium batch"),
            (100, "Large batch")
        ]
        
        # Test on both CPU and GPU for comparison
        devices = ['/CPU:0', '/GPU:0']
        
        for device in devices:
            print(f"\n{'='*20} {device} RESULTS {'='*20}")
            
            for space_name, space in spaces.items():
                print(f"\n📊 {space_name.upper().replace('_', ' ')} SPACE")
                print("-" * 40)
                
                for num_samples, description in test_configs:
                    print(f"\n{description} (num_samples={num_samples}):")
                    
                    try:
                        results = self.benchmark_fesbo_pattern(
                            space, num_samples, device, num_iterations=50  # Reduced for GPU memory
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

def provide_gpu_testing_guidance():
    """Provide guidance for GPU testing."""
    
    print("\n" + "="*60)
    print("🎯 GPU TESTING GUIDANCE")
    print("="*60)
    
    print("\n📋 WHAT TO LOOK FOR:")
    print("• Pure continuous spaces: Expect ~0% improvement (uses Sobol)")
    print("• Mixed spaces: Expect 10-30% improvement on GPU")
    print("• Pure discrete spaces: Expect 20-40% improvement on GPU")
    print("• GPU should show better scaling than CPU for larger batches")
    
    print("\n⚠️  POTENTIAL ISSUES:")
    print("• First GPU run may be slower (compilation)")
    print("• Small batches may show no improvement (overhead)")
    print("• Memory errors with very large batches")
    
    print("\n✅ SUCCESS INDICATORS:")
    print("• Mixed/discrete spaces show consistent improvement")
    print("• Larger batches scale better on GPU")
    print("• No memory errors or crashes")
    print("• Improvement is consistent across test runs")
    
    print("\n🔧 IF RESULTS ARE POOR:")
    print("• Check GPU utilization with nvidia-smi")
    print("• Verify GPU memory isn't maxed out")
    print("• Try reducing num_iterations if memory errors")
    print("• Ensure TensorFlow is using GPU (check logs)")

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
    print("• If results look good → implement the one-line change in fesbo.py")
    print("• If mixed/discrete spaces show improvement → you'll benefit!")
    print("• If pure continuous shows 0% → that's expected and normal")

if __name__ == "__main__":
    main()
