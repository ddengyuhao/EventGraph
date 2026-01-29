# run_inference.py

import argparse
import os
import json
import torch
import re
from tqdm import tqdm
from my_dataset import DATASET_REGISTRY
from methods import METHOD_REGISTRY

def parse_args():
    parser = argparse.ArgumentParser(description="EventGraph-LLM Inference Script")
    
    # Dataset & Paths
    parser.add_argument("--dataset", type=str, default="CinePile", choices=["VideoMME", "CinePile", "VRBench", "LVBench"])
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--output_dir", type=str, default="./results")
    
    # Model Configuration
    parser.add_argument("--method", type=str, default="EventGraph-LMM")
    parser.add_argument("--backbone", type=str, default="Video-LLaVA-7B", help="Backbone model name")
    parser.add_argument("--model_path", type=str, default=None, help="Local path to the LLM backbone (optional)")
    parser.add_argument("--clip_path", type=str, default=None, help="Local path to CLIP model (optional)")
    
    # Hyperparameters
    parser.add_argument("--token_budget", type=int, default=8192)
    parser.add_argument("--tau", type=float, default=30.0)
    parser.add_argument("--delta", type=float, default=0.65)
    
    # Parallel Processing
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for debugging")
    
    return parser.parse_args()

def extract_answer_from_text(text, options=None):
    """Parses LLM output to extract options like A, B, C, D."""
    if not text: return "C"
    text = text.strip()
    
    # Patterns to match answer
    patterns = [
        r'(?:answer|option)\s*(?:is|:)\s*[\(]?([A-D])[\)]?', # "The answer is A"
        r'(?:^|\s)[\(]?([A-D])[\)]?[\.\s]*$',                 # Ends with " A." or "(A)"
        r'^[\(]?([A-D])[\)]?[\.\s]'                           # Starts with "A."
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match: return match.group(1).upper()

    return "C" # Default fallback

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 [Chunk {args.chunk_idx}/{args.num_chunks}] Starting inference on {device}...")
    print(f"   Dataset: {args.dataset} | Method: {args.method} | Backbone: {args.backbone}")

    # 1. Load Dataset
    try:
        dataset_cls = DATASET_REGISTRY[args.dataset]
        dataset = dataset_cls(root_dir=args.data_root)
    except KeyError:
        print(f"❌ Error: Dataset '{args.dataset}' not registered.")
        return

    # 2. Chunking for Parallel Execution
    if args.max_samples is not None:
        dataset.samples = dataset.samples[:args.max_samples]
        print(f"⚠️ Debug: Truncated to {args.max_samples} samples.")

    total_samples = len(dataset)
    chunk_size = (total_samples + args.num_chunks - 1) // args.num_chunks
    start_idx = args.chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_samples)
    dataset.samples = dataset.samples[start_idx:end_idx]
    
    print(f"   Processing samples {start_idx} to {end_idx} (Count: {len(dataset)})")

    if len(dataset) == 0:
        print("✅ Chunk empty. Exiting.")
        return

    # 3. Load Model Backbone
    if args.backbone == "Video-LLaVA-7B":
        from models.video_llava_7b import VideoLLaVAWrapper
        model = VideoLLaVAWrapper(model_path=args.model_path)
    elif args.backbone == "Qwen2.5-VL-7B":
        from models.qwen2_5_vl import Qwen2_5_VLWrapper
        model = Qwen2_5_VLWrapper(model_path=args.model_path)
    elif args.backbone == "Qwen2-VL-72B":
        from models.qwen2_vl_72b import Qwen2_VL_72B_Wrapper
        model_path = args.model_path if args.model_path else "/root/hhq/models/Qwen/Qwen2___5-VL-72B-Instruct"
        model = Qwen2_VL_72B_Wrapper(model_path=model_path)
    else:
        print(f"❌ Unknown backbone: {args.backbone}")
        return

    # 4. Initialize Method
    method_cls = METHOD_REGISTRY[args.method]
    processor = method_cls(args, model)

    # 5. Inference Loop
    results = []
    
    for sample in tqdm(dataset, desc=f"GPU {args.chunk_idx}"):
        sample_id = sample.get('id', 'unknown')
        try:
            if not sample.get('video_path'):
                raise ValueError("Video path missing")

            pred_raw = processor.process_and_inference(
                sample['video_path'],
                sample['question'],
                sample.get('options', [])
            )
            
            pred_cleaned = extract_answer_from_text(pred_raw)
            
            results.append({
                "id": sample_id,
                "pred": pred_cleaned,
                "pred_raw": pred_raw,
                "gt": sample.get('answer', ''),
                "q": sample.get('question', '')
            })
            
            # Save intermediate results frequently
            if len(results) % 10 == 0:
                temp_path = os.path.join(args.output_dir, f"temp_{args.dataset}_{args.chunk_idx}.json")
                with open(temp_path, 'w') as f: json.dump(results, f)

        except Exception as e:
            print(f"❌ [Error] ID: {sample_id} | {e}")
            results.append({"id": sample_id, "error": str(e), "pred": "C", "gt": sample.get('answer', '')})

    # Final Save
    final_path = os.path.join(args.output_dir, f"{args.dataset}_{args.method}_chunk{args.chunk_idx}.json")
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✅ Chunk {args.chunk_idx} completed. Saved to {final_path}")

if __name__ == "__main__":
    main()