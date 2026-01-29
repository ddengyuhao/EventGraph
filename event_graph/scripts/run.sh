#!/bin/bash

# ================= Configuration =================
# Dataset: CinePile, VRBench, VideoMME
DATASET="CinePile" 

# Model: Qwen2.5-VL-7B, Qwen2-VL-72B, Video-LLaVA-7B
BACKBONE="Qwen2.5-VL-7B"

# Paths (Use absolute paths or relative to project root)
DATA_ROOT="./dataset"
OUTPUT_DIR="./experiments/${DATASET}_${BACKBONE}"

# Optional: Local Model Paths (Leave empty to use HuggingFace)
# MODEL_PATH="/path/to/your/Qwen2.5-VL-7B"
# CLIP_PATH="/path/to/your/clip-vit-large-patch14"

# Parameters
GPU_IDS=(0 1 2 3) # GPUs to use
NUM_GPUS=${#GPU_IDS[@]}
TOKEN_BUDGET=8192
# =================================================

echo "🚀 Starting Inference on $DATASET with $BACKBONE"
echo "   GPUs: ${GPU_IDS[*]} | Total Chunks: $NUM_GPUS"

mkdir -p "$OUTPUT_DIR"

pids=()

for ((i=0; i<NUM_GPUS; i++)); do
    GPU_ID=${GPU_IDS[$i]}
    
    cmd="python run_inference.py \
        --dataset $DATASET \
        --backbone $BACKBONE \
        --data_root $DATA_ROOT \
        --output_dir $OUTPUT_DIR \
        --num_chunks $NUM_GPUS \
        --chunk_idx $i \
        --token_budget $TOKEN_BUDGET"
    
    # Add optional paths if set
    [ ! -z "$MODEL_PATH" ] && cmd="$cmd --model_path $MODEL_PATH"
    [ ! -z "$CLIP_PATH" ] && cmd="$cmd --clip_path $CLIP_PATH"

    echo "   > Running Chunk $i on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID $cmd > "$OUTPUT_DIR/log_gpu${GPU_ID}.txt" 2>&1 &
    
    pids+=($!)
done

# Wait for all
for pid in "${pids[@]}"; do
    wait $pid
done

echo "✅ Inference Completed. Merging results..."

# Merge Script
python -c "
import json, glob, os
files = glob.glob('$OUTPUT_DIR/*_chunk*.json')
all_data = []
for f in files:
    try:
        with open(f) as fd: all_data.extend(json.load(fd))
    except: print(f'Skipping broken file {f}')
    
out_path = os.path.join('$OUTPUT_DIR', 'final_merged.json')
with open(out_path, 'w') as f:
    json.dump(all_data, f, indent=2)
print(f'Merged {len(all_data)} records to {out_path}')
"