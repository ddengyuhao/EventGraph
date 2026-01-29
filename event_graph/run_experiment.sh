#!/bin/bash

# Configuration
# DATASET="CinePile"
# DATASET="VRBench"
# DATASET="VideoMME"
DATASET="LVBench"


METHOD="EventGraph-LMM"
# METHOD="No-Compression"

BACKBONE="Qwen2.5-VL-7B" # Or "LLaVA-NeXT-Video-34B"
# BACKBONE="Video-LLaVA-7B" # Or "LLaVA-NeXT-Video-34B"

MAX_SAMPLES=100 # Change to 100 for a quick subset test, or remove for full run
OUTPUT_DIR="./results_$DATASET"
# 如果跑 Qwen2.5-VL
Token_Budget=8192

# 如果跑 Video-LLaVA-7B
# Token_Budget=4096

# 如果跑 Video-LLaVA-34B
# Token_Budget=6144

# Array to store PIDs
pids=()

echo "🚀 Starting 4-GPU Inference on $DATASET..."

# Loop for 4 GPUs (0, 1, 2, 3)
for GPU_ID in 0 1 2 3
do
    echo "  > Launching Chunk $GPU_ID on GPU $GPU_ID..."
    
    # Calculate log name
    LOG_FILE="log_gpu${GPU_ID}.txt"
    
    # Run in background (&)
    # Note: We pass --max_samples to limit the TOTAL dataset before chunking
    CUDA_VISIBLE_DEVICES=$GPU_ID python run_inference.py \
        --dataset $DATASET \
        --method $METHOD \
        --backbone $BACKBONE \
        --num_chunks 4 \
        --chunk_idx $GPU_ID \
        --max_samples $MAX_SAMPLES \
        --output_dir $OUTPUT_DIR \
        --token_budget $Token_Budget \
        > $LOG_FILE 2>&1 &
        
    # Store PID
    pids+=($!)
done

# Wait for all processes to finish
echo "⏳ Waiting for all processes to finish..."
for pid in "${pids[@]}"; do
    wait $pid
done

echo "✅ All chunks completed. Merging results..."

# Simple Python one-liner to merge JSONs
python -c "
import json, glob, os
files = glob.glob('$OUTPUT_DIR/${DATASET}_${METHOD}_chunk*.json')
all_data = []
for f in files:
    with open(f) as fd: all_data.extend(json.load(fd))
with open('$OUTPUT_DIR/FINAL_MERGED.json', 'w') as f:
    json.dump(all_data, f, indent=4)
print(f'Merged {len(all_data)} records.')
"