#!/bin/bash
NUM_RUNS=2
GPU_IDS=( 0 1 2 3 4 5 6 7 ) 
NUM_GPUS=${#GPU_IDS[@]}
counter=0

DATASETS=( 'fmow' )
SEEDS=( 42 1234 )
ALPHA=('0.0' '0.5' '1.0' '5.0' '10.0' '100.0')
ALGORITHMS=( "ERM-aug" )

for dataset in "${DATASETS[@]}"; do
for algorithm in "${ALGORITHMS[@]}"; do
for seed in "${SEEDS[@]}"; do

	 # Get GPU id.
	 gpu_idx=$((counter % $NUM_GPUS))
	 gpu_id=${GPU_IDS[$gpu_idx]}
	 
     cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python run_expt.py '--remote' 'False' '--dataset' ${dataset} '--n_epochs' 60 --resume\
	 '--root_dir' '/home/sgarg2/data' '--seed' ${seed} '--transform' 'image_none' '--algorithm'  ${algorithm} --progress_bar"
	 
     echo $cmd
	 eval ${cmd} &

	 counter=$((counter+1))
	 if ! ((counter % NUM_RUNS)); then
		  wait
	 fi
done
done
done


