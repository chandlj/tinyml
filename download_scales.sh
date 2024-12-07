#!/bin/bash

# Create directory if it doesn't exist
mkdir -p ./smoothquant/act_scales

# Array of model sizes
opt_sizes=("125m" "1.3b" "2.7b" "6.7b" "13b" "30b" "66b")
llama_sizes=("7b" "13b" "70b")

# Download OPT model scales
for size in "${opt_sizes[@]}"; do
    file="./smoothquant/act_scales/opt-${size}.pt"
    if [ ! -f "$file" ]; then
        echo "Downloading scales for OPT-${size}..."
        wget -P ./smoothquant/act_scales/ \
            "https://huggingface.co/mit-han-lab/smoothquant-scales/resolve/main/opt-${size}.pt"
    else
        echo "Scales for OPT-${size} already exist, skipping..."
    fi
done

# Download Llama-2 model scales 
for size in "${llama_sizes[@]}"; do
    file="./smoothquant/act_scales/llama-2-${size}.pt"
    if [ ! -f "$file" ]; then
        echo "Downloading scales for Llama-2-${size}..."
        wget -P ./smoothquant/act_scales/ \
            "https://huggingface.co/mit-han-lab/smoothquant-scales/resolve/main/llama-2-${size}.pt"
    else
        echo "Scales for Llama-2-${size} already exist, skipping..."
    fi
done

echo "Download complete!"