#!/bin/bash

#SBATCH -J training_dpo
#SBATCH -o training_dpo_%A_%a.out
#SBATCH -e training_dpo_%A_%a.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem=16G
#SBATCH --gres=gpu:H100:1
#SBATCH --array=0-0

#hyperparameters
batch_sizes=(8)
epochs=(20)
lambda_vals=(1000)
step_sizes=(20)
thresholds=(0.00001)

total_combinations=$((${#batch_sizes[@]} * ${#epochs[@]} * ${#lambda_vals[@]} * ${#step_sizes[@]} * ${#thresholds[@]}))

echo "Total combinations: $total_combinations"

# Check if SLURM_ARRAY_TASK_ID is within range
if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
    echo "Array task ID out of range! SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

batch_size_index=$((SLURM_ARRAY_TASK_ID % ${#batch_sizes[@]}))
epochs_index=$(( (SLURM_ARRAY_TASK_ID / ${#batch_sizes[@]}) % ${#epochs[@]} ))
lambda_index=$(( (SLURM_ARRAY_TASK_ID / (${#batch_sizes[@]} * ${#epochs[@]})) % ${#lambda_vals[@]} ))
step_size_index=$(( (SLURM_ARRAY_TASK_ID / (${#batch_sizes[@]} * ${#epochs[@]} * ${#lambda_vals[@]})) % ${#step_sizes[@]} ))
threshold_index=$(( (SLURM_ARRAY_TASK_ID / (${#batch_sizes[@]} * ${#epochs[@]} * ${#lambda_vals[@]} * ${#step_sizes[@]})) % ${#thresholds[@]} ))

batch_size=${batch_sizes[$batch_size_index]}
epochs=${epochs[$epochs_index]}
lambda_penalty=${lambda_vals[$lambda_index]}
step_size=${step_sizes[$step_size_index]}
threshold=${thresholds[$threshold_index]}

source activate base
conda activate dpo

apptainer exec --nv /home/sagnikg/containers/miniconda3_latest.sif bash -c "
    source activate base
    conda activate dpo
    env WANDB_MODE=offline python dpo-penalty-token.py \
        --epochs $epochs \
        --batch_size $batch_size \
        --lambda_penalty $lambda_penalty \
        --step_size $step_size \
        --threshold $threshold
"