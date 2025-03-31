#!/bin/bash 

source ~/.bashrc
conda activate cs726_a3

# Define the range for beam width (W) and number of Medusa heads (S)
beam_widths=(2 5 10)
medusa_heads=(2 5)

output_dir="./Results/1.3_Staring_into_Medusa_Heads/"

# Function to check GPU with least memory usage
get_gpu_with_least_memory() {
    # Run nvidia-smi and extract GPU memory usage details (index, used, free memory)
    gpu_info=$(nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader,nounits)

    # Initialize variables to track GPU with least memory usage
    min_memory_used=99999999
    gpu_id_with_least_memory=-1

    while IFS=, read -r gpu_id memory_free memory_used; do
        if [ "$memory_used" -lt "$min_memory_used" ]; then
            min_memory_used=$memory_used
            gpu_id_with_least_memory=$gpu_id
        fi
    done <<< "$gpu_info"

    echo $gpu_id_with_least_memory
}

# Run Single Head Medusa
echo "Running Single Head Medusa"
gpu_id_with_least_memory=$(get_gpu_with_least_memory)
echo "Using GPU: ${gpu_id_with_least_memory}"
CUDA_VISIBLE_DEVICES=${gpu_id_with_least_memory} python task2.py --hf-token $HF_TOKEN --decoding-strategy "single-head" > ${output_dir}/single_head_decoding.txt 2>&1

# Run Multiple Head Medusa
echo "Running Multiple Head Medusa"
for W in "${beam_widths[@]}"; do
    for S in "${medusa_heads[@]}"; do
        echo "Running with W=$W and S=$S"
        gpu_id_with_least_memory=$(get_gpu_with_least_memory)
        echo "Using GPU: ${gpu_id_with_least_memory}"
        CUDA_VISIBLE_DEVICES=${gpu_id_with_least_memory} python task2.py --hf-token $HF_TOKEN --decoding-strategy "multi-head" --beam-width $W --use-no-medusa-heads $S > ${output_dir}/multiple_${S}_head_${W}_beam_decoding.txt 2>&1
    done
done