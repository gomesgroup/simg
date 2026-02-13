#!/bin/bash

targets=("atomization" "dipole" "gap" "homo")
fractions=(0.2 0.4 0.6 0.8 1.0)
log_file="experiment_log.txt"

exec 1> >(tee -a "$log_file") 2>&1
echo "Starting experiments at $(date)"

for target in "${targets[@]}"; do
    for fraction in "${fractions[@]}"; do
        echo "Starting target: $target, fraction: $fraction"

        CUDA_VISIBLE_DEVICES=0 python train.py --target "$target" --training_fraction "$fraction" --attempt 0 &
        CUDA_VISIBLE_DEVICES=1 python train.py --target "$target" --training_fraction "$fraction" --attempt 1 &
        CUDA_VISIBLE_DEVICES=2 python train.py --target "$target" --training_fraction "$fraction" --attempt 2 &

        wait

        echo "Completed target: $target, fraction: $fraction"
    done
done

echo "Completed all experiments at $(date)"