#!/bin/bash

output_dir="./Results/1.1_Introduction_to_LLM_Decoding_Techniques"
mkdir -p "$output_dir"

source ~/.bashrc
conda activate cs726_a3
echo "Running task0.py with greedy decoding"
CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token $HF_TOKEN --decoding-strategy "greedy" > ${output_dir}/greedy_outputs.txt 2>&1
echo "Greedy Decoding Completed\n"


tau_values=(0.5 0.6 0.7 0.8 0.9)
for tau in "${tau_values[@]}"
do
    echo "Running task0.py with tau=$tau..."
    CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token $HF_TOKEN --decoding-strategy "random" --tau $tau > ${output_dir}/tau_${tau}_outputs.txt 2>&1
done
echo "Random Sampling with Temperature Scaling completed.\n"


echo "Running task0.py with top-k sampling"
top_k_values=(5 6 7 8 9 10)
for top_k_val in "${top_k_values[@]}"
do
    echo "Running task0.py with topk=$top_k_val..."
    CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token $HF_TOKEN --decoding-strategy "topk" --k $top_k_val > ${output_dir}/topk_${tau}.txt 2>&1
done
echo "Top-k Sampling Completed\n"

echo "Running task0.py with Nucleus Sampling"
p_values=(0.5 0.6 0.7 0.8 0.9)
for p in "${p_values[@]}"
do
    echo "Running task0.py with p=$p..."
    CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token $HF_TOKEN --decoding-strategy "nucleus" --p $p > ${output_dir}/nucleus_sampling_${p}_outputs.txt 2>&1
done
echo "Nucleus Sampling Completed\n"

echo "Plotting the Metrics"
conda activate myenv
python plot.py
echo "Results stored in ./Results/"
