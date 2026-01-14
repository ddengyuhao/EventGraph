# /root/ICML2026/event_graph/run_inference.py
import argparse
import os
import json
import torch
from tqdm import tqdm
from my_datasets import DATASET_REGISTRY
from code import METHOD_REGISTRY

def parse_args():
    parser = argparse.ArgumentParser(description="EventGraph-LLM Experiments")
    parser.add_argument("--dataset", type=str, default="CinePile", choices=["VideoMME", "CinePile"])
    parser.add_argument("--method", type=str, default="EventGraph-LMM")
    parser.add_argument("--backbone", type=str, default="Video-LLaVA-7B")
    parser.add_argument("--data_root", type=str, default="/root/ICML2026/dataset")
    parser.add_argument("--token_budget", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./result")
    
    # Efficient Parallelism Args
    parser.add_argument("--num_chunks", type=int, default=1, help="Total GPU count")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Current GPU ID")
    
    # Testing Subset Args
    parser.add_argument("--max_samples", type=int, default=None, help="Debug: Run only N samples")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Device Setup for A100s
    # If running via launch script, CUDA_VISIBLE_DEVICES will handle this, 
    # but explicit device map is safer.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Process {args.chunk_idx}/{args.num_chunks} starting on {device}")

    # 2. Load Dataset
    print(f"📂 Loading Dataset: {args.dataset}")
    dataset_cls = DATASET_REGISTRY[args.dataset]
    dataset = dataset_cls(root_dir=args.data_root)
    
    # Apply Subset (Max Samples) BEFORE Chunking
    if args.max_samples is not None:
        dataset.samples = dataset.samples[:args.max_samples]
        print(f"⚠️ DEBUG MODE: Truncated to first {args.max_samples} samples.")

    # Apply Chunking (Split work among 4 GPUs)
    total_samples = len(dataset)
    chunk_size = (total_samples + args.num_chunks - 1) // args.num_chunks
    start_idx = args.chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_samples)
    
    dataset.samples = dataset.samples[start_idx:end_idx]
    print(f"🔄 Chunk {args.chunk_idx}: Processing samples {start_idx} to {end_idx} (Count: {len(dataset)})")

    if len(dataset) == 0:
        print("✅ Chunk is empty, exiting.")
        return

    # 3. Load Model (Lazy load inside main to avoid multi-process fork issues)
    # Note: Import Wrapper here
    if args.backbone == "Video-LLaVA-7B":
        from models.video_llava_7b import VideoLLaVAWrapper
        model = VideoLLaVAWrapper() # Internally handles .to(device)
    elif "34B" in args.backbone:
        from models.llava_next_34b import LLaVANext34BWrapper
        model = LLaVANext34BWrapper()

    # 4. Initialize Method
    method_cls = METHOD_REGISTRY[args.method]
    processor = method_cls(args, model)

    # 5. Inference Loop
    results = []
    os.makedirs(args.output_dir, exist_ok=True)
    
    for sample in tqdm(dataset, desc=f"GPU {args.chunk_idx}"):
        try:
            # CinePile returns dict with 'video_path', 'question', 'options'
            pred = processor.process_and_inference(
                sample['video_path'],
                sample['question'],
                sample.get('options', [])
            )
            
            results.append({
                "id": sample['id'],
                "pred": pred,
                "gt": sample.get('answer', ''),
                "q": sample['question']
            })
            
            # Incremental save (Crucial for long runs!)
            if len(results) % 5 == 0:
                temp_path = os.path.join(args.output_dir, f"temp_{args.dataset}_{args.chunk_idx}.json")
                with open(temp_path, 'w') as f: json.dump(results, f)

        except Exception as e:
            print(f"❌ Error {sample['id']}: {e}")

    # Final Save
    final_path = os.path.join(args.output_dir, f"{args.dataset}_{args.method}_chunk{args.chunk_idx}.json")
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✅ Saved to {final_path}")

if __name__ == "__main__":
    main()