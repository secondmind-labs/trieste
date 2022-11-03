#!/bin/bash

# Activate virtual environment
. venv-trieste/bin/activate

# Disable GPUs to ensure running on CPUs
export CUDA_VISIBLE_DEVICES=1

# Parameters
obj_dims=("4" "6" "8")
objectives=("eq")
batch_size=5
num_batches=3
num_initial_designs=1000
num_mcei_samples=1000
num_seeds=5
num_sobol=100
ftol=-10
gtol=-6

tol_factors=("0" "-4" "-8")
nums_sobol=("10" "50" "100")
nums_initial=("10" "100" "1000")
nums_parallel=("1" "10" "100")


for obj_dim in ${obj_dims[@]}
do

    # =============================================================================
    # Random acquisition
    # =============================================================================

    for obj in ${objectives[@]}
    do
        for seed in $( seq 0 $(($num_seeds-1)) )
        do
            # Run random acquisition experiment
            python /scratches/cblgpu07/em626/trieste/scripts/run.py \
              random \
              $obj \
              -batch_size=$batch_size \
              -num_batches=$(($num_batches*$obj_dim)) \
              --search_seed=$seed \
              --objective_seed=$seed \
              --num_initial_designs=$num_initial_designs \
              --objective_dimension=$obj_dim \
              --gtol=$gtol \
              --ftol=$ftol
        done
    done


    # =============================================================================
    # Single-point EI acquisition
    # =============================================================================

    for obj in ${objectives[@]}
    do
        for seed in $( seq 0 $(($num_seeds-1)) )
        do
            for tol_factor in ${tol_factors[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  ei \
                  $obj \
                  -batch_size=$batch_size \
                  -num_batches=$(($num_batches*$obj_dim)) \
                  --search_seed=$seed \
                  --objective_seed=$seed \
                  --num_initial_designs=$num_initial_designs \
                  --gtol=$(($gtol+$tol_factor)) \
                  --ftol=$(($ftol+$tol_factor)) \
                  --objective_dimension=$obj_dim
            done

            for num_initial in ${nums_initial[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  ei \
                  $obj \
                  -batch_size=$batch_size \
                  -num_batches=$(($num_batches*$obj_dim)) \
                  --search_seed=$seed \
                  --objective_seed=$seed \
                  --num_initial_designs=$num_initial \
                  --gtol=$gtol \
                  --ftol=$ftol \
                  --objective_dimension=$obj_dim
            done

            for num_parallel in ${nums_parallel[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  ei \
                  $obj \
                  -batch_size=$batch_size \
                  -num_batches=$(($num_batches*$obj_dim)) \
                  --search_seed=$seed \
                  --objective_seed=$seed \
                  --num_optimization_runs=$num_parallel \
                  --gtol=$gtol \
                  --ftol=$ftol \
                  --objective_dimension=$obj_dim
            done
        done
    done


    # =============================================================================
    # Single-point MCEI acquisition
    # =============================================================================

    for obj in ${objectives[@]}
    do
        for seed in $( seq 0 $(($num_seeds-1)) )
        do
            for tol_factor in ${tol_factors[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  mcei \
                  $obj \
                  -batch_size=1 \
                  -num_batches=$(($batch_size*$(($num_batches*$obj_dim)))) \
                  --objective_seed=$seed \
                  --search_seed=$seed \
                  --num_mcei_samples=$num_mcei_samples \
                  --num_initial_designs=$num_initial_designs \
                  --gtol=$(($gtol+$tol_factor)) \
                  --ftol=$(($ftol+$tol_factor)) \
                  --objective_dimension=$obj_dim
            done

            for num_initial in ${nums_initial[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  mcei \
                  $obj \
                  -batch_size=1 \
                  -num_batches=$(($batch_size*$(($num_batches*$obj_dim)))) \
                  --objective_seed=$seed \
                  --search_seed=$seed \
                  --num_mcei_samples=$num_mcei_samples \
                  --num_initial_designs=$num_initial  \
                  --gtol=$gtol \
                  --ftol=$ftol \
                  --objective_dimension=$obj_dim
            done

            for num_parallel in ${nums_parallel[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  mcei \
                  $obj \
                  -batch_size=1 \
                  -num_batches=$(($batch_size*$(($num_batches*$obj_dim)))) \
                  --objective_seed=$seed \
                  --search_seed=$seed \
                  --num_mcei_samples=$num_mcei_samples \
                  --num_optimization_runs=$num_parallel \
                  --gtol=$gtol \
                  --ftol=$ftol \
                  --objective_dimension=$obj_dim
            done
        done
    done


    # =============================================================================
    # Batch MCEI acquisition
    # =============================================================================

    for obj in ${objectives[@]}
    do
        for seed in $( seq 0 $(($num_seeds-1)) )
        do
            for tol_factor in ${tol_factors[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  mcei \
                  $obj \
                  -batch_size=$batch_size \
                  -num_batches=$(($num_batches*$obj_dim)) \
                  --search_seed=$seed \
                  --objective_seed=$seed \
                  --num_initial_designs=$num_initial_designs \
                  --num_mcei_samples=$num_mcei_samples \
                  --gtol=$(($gtol+$tol_factor)) \
                  --ftol=$(($ftol+$tol_factor)) \
                  --objective_dimension=$obj_dim
            done

            for num_initial in ${nums_initial[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  mcei \
                  $obj \
                  -batch_size=$batch_size \
                  -num_batches=$(($num_batches*$obj_dim)) \
                  --search_seed=$seed \
                  --objective_seed=$seed \
                  --num_initial_designs=$num_initial \
                  --num_mcei_samples=$num_mcei_samples \
                  --gtol=$gtol \
                  --ftol=$ftol \
                  --objective_dimension=$obj_dim
            done

            for num_parallel in ${nums_parallel[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  mcei \
                  $obj \
                  -batch_size=$batch_size \
                  -num_batches=$(($num_batches*$obj_dim)) \
                  --search_seed=$seed \
                  --objective_seed=$seed \
                  --num_mcei_samples=$num_mcei_samples \
                  --num_optimization_runs=$num_parallel \
                  --gtol=$gtol \
                  --ftol=$ftol \
                  --objective_dimension=$obj_dim
            done
        done
    done


    # =============================================================================
    # Batch Thompson sampling acquisition
    # =============================================================================

    for obj in ${objectives[@]}
    do
        for seed in $( seq 0 $(($num_seeds-1)) )
        do
            for tol_factor in ${tol_factors[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  thompson \
                  $obj \
                  -batch_size=$batch_size \
                  -num_batches=$(($num_batches*$obj_dim)) \
                  --search_seed=$seed \
                  --objective_seed=$seed \
                  --num_initial_designs=$num_initial_designs \
                  --gtol=$(($gtol+$tol_factor)) \
                  --ftol=$(($ftol+$tol_factor)) \
                  --objective_dimension=$obj_dim
            done

            for num_initial in ${nums_initial[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  thompson \
                  $obj \
                  -batch_size=$batch_size \
                  -num_batches=$(($num_batches*$obj_dim)) \
                  --search_seed=$seed \
                  --objective_seed=$seed \
                  --num_initial_designs=$num_initial \
                  --gtol=$gtol \
                  --ftol=$ftol \
                  --objective_dimension=$obj_dim
            done

            for num_parallel in ${nums_parallel[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  thompson \
                  $obj \
                  -batch_size=$batch_size \
                  -num_batches=$(($num_batches*$obj_dim)) \
                  --search_seed=$seed \
                  --objective_seed=$seed \
                  --num_optimization_runs=$num_parallel \
                  --gtol=$gtol \
                  --ftol=$ftol \
                  --objective_dimension=$obj_dim
            done
        done
    done


    # =============================================================================
    # Batch Multipoint Expected Improvement
    # =============================================================================

    for obj in ${objectives[@]}
    do
        for seed in $( seq 0 $(($num_seeds-1)) )
        do
            for tol_factor in ${tol_factors[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  qei \
                  $obj \
                  -batch_size=$batch_size \
                  -num_batches=$(($num_batches*$obj_dim)) \
                  --search_seed=$seed \
                  --objective_seed=$seed \
                  --num_initial_designs=$num_initial_designs \
                  --gtol=$(($gtol+$tol_factor)) \
                  --ftol=$(($ftol+$tol_factor)) \
                  --objective_dimension=$obj_dim
            done

            for num_initial in ${nums_initial[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  qei \
                  $obj \
                  -batch_size=$batch_size \
                  -num_batches=$(($num_batches*$obj_dim)) \
                  --search_seed=$seed \
                  --objective_seed=$seed \
                  --num_initial_designs=$num_initial \
                  --gtol=$gtol \
                  --ftol=$ftol \
                  --qei_sample_size=$num_sobol \
                  --objective_dimension=$obj_dim
            done

            for num_parallel in ${nums_parallel[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  qei \
                  $obj \
                  -batch_size=$batch_size \
                  -num_batches=$(($num_batches*$obj_dim)) \
                  --search_seed=$seed \
                  --objective_seed=$seed \
                  --num_optimization_runs=$num_parallel \
                  --gtol=$gtol \
                  --ftol=$ftol \
                  --qei_sample_size=$num_sobol \
                  --objective_dimension=$obj_dim
            done

            for num_sobol in ${nums_sobol[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                  qei \
                  $obj \
                  -batch_size=$batch_size \
                  -num_batches=$(($num_batches*$obj_dim)) \
                  --search_seed=$seed \
                  --objective_seed=$seed \
                  --num_initial_designs=$num_initial_designs \
                  --gtol=$gtol \
                  --ftol=$ftol \
                  --qei_sample_size=$num_sobol \
                  --objective_dimension=$obj_dim
            done
        done
    done
done