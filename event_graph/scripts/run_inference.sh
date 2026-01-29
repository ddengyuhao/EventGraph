#!/bin/bash

# Default Settings
DATASET=${1:-"LVBench"}       # First arg: Dataset
GPU_COUNT=${2:-4}             # Second arg: Number of GPUs
BACKBONE="Qwen2.5-VL-7B"      # Configure backbone here
METHOD="EventGraph-LMM"

# Paths (Ideally, set these as environment variables)
DATA_ROOT="/root/icml2026/dataset"
MODEL_PATH="/root/hhq/models/Qwen/Qwen2.5-VL-7B-Instruct"
CLIP_PATH="/root/hhq/models/clip-vit-large-patch14"
OUTPUT_DIR="./results/${DATASET}_${BACKBONE}"

# Params
TOKEN_BUDGET=8192
MAX_SAMPLES=100  # Set to empty string "" for full run

echo "🚀 Starting Inference: $DATASET with $BACKBONE ($METHOD)"
echo "   GPUs: $GPU_COUNT | Budget: $TOKEN_BUDGET"

pids=()

for ((i=0; i<GPU_COUNT; i++)); do
    echo "   > Launching chunk $i on GPU $i..."
    
    CUDA_VISIBLE_DEVICES=$i python run_inference.py \
        --dataset "$DATASET" \
        --method "$METHOD" \
        --backbone "$BACKBONE" \
        --data_root "$DATA_ROOT" \
        --model_path "$MODEL_PATH" \
        --clip_path "$CLIP_PATH" \
        --num_chunks "$GPU_COUNT" \
        --chunk_idx "$i" \
        --output_dir "$OUTPUT_DIR" \
        --token_budget "$TOKEN_BUDGET" \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
        > "${OUTPUT_DIR}/log_gpu${i}.txt" 2>&1 &
        
    pids+=($!)
done

# Wait
for pid in "${pids[@]}"; do
    wait $pid
done

echo "✅ All chunks finished. Merging..."

python -c "
import json, glob, os
files = glob.glob('$OUTPUT_DIR/*_chunk*.json')
all_data = []
for f in files:
    with open(f) as fd: all_data.extend(json.load(fd))
out_path = os.path.join('$OUTPUT_DIR', 'final_merged.json')
with open(out_path, 'w') as f:
    json.dump(all_data, f, indent=4)
print(f'Merged {len(all_data)} records to {out_path}')
"