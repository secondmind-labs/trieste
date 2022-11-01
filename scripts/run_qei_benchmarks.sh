#!/bin/bash

# Activate virtual environment
. venv-trieste/bin/activate

# Disable GPUs to ensure running on CPUs
export CUDA_VISIBLE_DEVICES=1

# Parameters
obj_dims=(4 6 8)
objectives=("eq")
batch_size=5
num_batches=10
num_initial_designs=10000
num_mcei_samples=1000
num_objective_seeds=0
search_seed=0
num_sobol=100
ftol=0.0000000001
gtol=0.000001

tol_factors=(1. 0.01 0.0001 0.0000001)
nums_sobol=(10 20 50)
nums_initial=(10 100 1000 10000)
nums_parallel=(1 5 10 100)


for obj_dim in ${obj_dims[@]}
do

    # =============================================================================
    # Random acquisition
    # =============================================================================

    for obj in ${objectives[@]}
    do
        for objective_seed in $( seq 0 $num_objective_seeds )
        do
            # Run random acquisition experiment
            python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                random \
                $obj \
                -batch_size=$batch_size \
                -num_batches=$num_batches \
                --search_seed=$search_seed \
                --objective_seed=$objective_seed \
                --num_initial_designs=$num_initial_designs \
                --objective_dimension=$obj_dim
        done
    done


    # =============================================================================
    # Single-point EI acquisition
    # =============================================================================

    for obj in ${objectives[@]}
    do
        for objective_seed in $( seq 0 $num_objective_seeds )
        do
            for tol_factor in ${tol_factors[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                    ei \
                    $obj \
                    -batch_size=$batch_size \
                    -num_batches=$num_batches \
                    --search_seed=$search_seed \
                    --objective_seed=$objective_seed \
                    --num_initial_designs=$num_initial_designs \
                    --gtol=$(echo "$gtol * $tol_factor" | bc) \
                    --ftol=$(echo "$ftol * $tol_factor" | bc) \
                    --objective_dimension=$obj_dim
            done

            for num_initial in ${nums_initial[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                    ei \
                    $obj \
                    -batch_size=$batch_size \
                    -num_batches=$num_batches \
                    --search_seed=$search_seed \
                    --objective_seed=$objective_seed \
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
                    -num_batches=$num_batches \
                    --search_seed=$search_seed \
                    --objective_seed=$objective_seed \
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
        for objective_seed in $( seq 0 $num_objective_seeds )
        do
            for tol_factor in ${tol_factors[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                    mcei \
                    $obj \
                    -batch_size=1 \
                    -num_batches=$(($batch_size*$num_batches)) \
                    --objective_seed=$objective_seed \
                    --search_seed=$search_seed \
                    --num_mcei_samples=$num_mcei_samples \
                    --num_initial_designs=$num_initial_designs \
                    --gtol=$(echo "$gtol * $tol_factor" | bc) \
                    --ftol=$(echo "$ftol * $tol_factor" | bc) \
                    --objective_dimension=$obj_dim
            done

            for num_initial in ${nums_initial[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                    mcei \
                    $obj \
                    -batch_size=1 \
                    -num_batches=$(($batch_size*$num_batches)) \
                    --objective_seed=$objective_seed \
                    --search_seed=$search_seed \
                    --num_mcei_samples=$num_mcei_samples \
                    --num_initial_designs=$num_initial \
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
                    -num_batches=$(($batch_size*$num_batches)) \
                    --objective_seed=$objective_seed \
                    --search_seed=$search_seed \
                    --num_mcei_samples=$num_mcei_samples \
                    --num_optimization_runs=$num_parallel 
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
        for objective_seed in $( seq 0 $num_objective_seeds )
        do
            for tol_factor in ${tol_factors[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                    mcei \
                    $obj \
                    -batch_size=$batch_size \
                    -num_batches=$num_batches \
                    --search_seed=$search_seed \
                    --objective_seed=$objective_seed \
                    --num_initial_designs=$num_initial_designs \
                    --num_mcei_samples=$num_mcei_samples \
                    --gtol=$(echo "$gtol * $tol_factor" | bc) \
                    --ftol=$(echo "$ftol * $tol_factor" | bc) \
                    --objective_dimension=$obj_dim
            done

            for num_initial in ${nums_initial[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                    mcei \
                    $obj \
                    -batch_size=$batch_size \
                    -num_batches=$num_batches \
                    --search_seed=$search_seed \
                    --objective_seed=$objective_seed \
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
                    -num_batches=$num_batches \
                    --search_seed=$search_seed \
                    --objective_seed=$objective_seed \
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
        for objective_seed in $( seq 0 $num_objective_seeds )
        do
            for tol_factor in ${tol_factors[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                    thompson \
                    $obj \
                    -batch_size=$batch_size \
                    -num_batches=$num_batches \
                    --search_seed=$search_seed \
                    --objective_seed=$objective_seed \
                    --num_initial_designs=$num_initial_designs \
                    --gtol=$(echo "$gtol * $tol_factor" | bc) \
                    --ftol=$(echo "$ftol * $tol_factor" | bc)
            done

            for num_initial in ${nums_initial[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                    thompson \
                    $obj \
                    -batch_size=$batch_size \
                    -num_batches=$num_batches \
                    --search_seed=$search_seed \
                    --objective_seed=$objective_seed \
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
                    -num_batches=$num_batches \
                    --search_seed=$search_seed \
                    --objective_seed=$objective_seed \
                    --num_optimisation_runs=$num_parallel \
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
        for objective_seed in $( seq 0 $num_objective_seeds )
        do
            for tol_factor in ${tol_factors[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                    qei \
                    $obj \
                    -batch_size=$batch_size \
                    -num_batches=$num_batches \
                    --search_seed=$search_seed \
                    --objective_seed=$objective_seed \
                    --num_initial_designs=$num_initial_designs \
                    --gtol=$(echo "$gtol * $tol_factor" | bc) \
                    --ftol=$(echo "$ftol * $tol_factor" | bc) \
                    --objective_dimension=$obj_dim
            done

            for num_initial in ${nums_initial[@]}
            do
                python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                    qei \
                    $obj \
                    -batch_size=$batch_size \
                    -num_batches=$num_batches \
                    --search_seed=$search_seed \
                    --objective_seed=$objective_seed \
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
                    -num_batches=$num_batches \
                    --search_seed=$search_seed \
                    --objective_seed=$objective_seed \
                    --num_optimisation_runs=$num_parallel \
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
                    -num_batches=$num_batches \
                    --search_seed=$search_seed \
                    --objective_seed=$objective_seed \
                    --num_initial_designs=$num_initial_designs \
                    --gtol=$gtol \
                    --ftol=$ftol \
                    --qei_sample_size=$num_sobol \
                    --objective_dimension=$obj_dim
            done
        done
    done
done