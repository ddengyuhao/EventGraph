import argparse
import os
import json
import torch
import re
from tqdm import tqdm
from event_graph.data import DATASET_REGISTRY
from event_graph.methods import METHOD_REGISTRY

def parse_args():
    parser = argparse.ArgumentParser(description="EventGraph-LLM Inference")
    
    # --- Paths & Dataset ---
    parser.add_argument("--dataset", type=str, required=True, choices=["VideoMME", "CinePile", "VRBench"], help="Target dataset")
    parser.add_argument("--data_root", type=str, default="./dataset", help="Root directory for datasets")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    
    # --- Model Config ---
    parser.add_argument("--method", type=str, default="EventGraph-LMM")
    parser.add_argument("--backbone", type=str, default="Qwen2.5-VL-7B", help="Model backbone")
    parser.add_argument("--model_path", type=str, default=None, help="Path to local LLM checkpoint")
    parser.add_argument("--clip_path", type=str, default=None, help="Path to local CLIP model")
    
    # --- Hyperparameters ---
    parser.add_argument("--token_budget", type=int, default=8192)
    parser.add_argument("--tau", type=float, default=30.0, help="Similarity threshold for graph")
    parser.add_argument("--delta", type=float, default=0.65, help="Temporal linking threshold")
    
    # --- Distributed ---
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None, help="Debug: limit sample count")
    
    return parser.parse_args()

def extract_answer_from_text(text):
    if not text: return "C"
    text = text.strip()
    # Matches: "The answer is A", "(A)", "A."
    patterns = [
        r'(?:answer|option)\s*(?:is|:)\s*[\(]?([A-D])[\)]?', 
        r'(?:^|\s)[\(]?([A-D])[\)]?[\.\s]*$', 
        r'^[\(]?([A-D])[\)]?[\.\s]'
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match: return match.group(1).upper()
    return "C"

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"🚀 [Chunk {args.chunk_idx}/{args.num_chunks}] Inference: {args.dataset} | {args.backbone}")

    # 1. Load Dataset
    if args.dataset not in DATASET_REGISTRY:
        print(f"❌ Dataset {args.dataset} not supported.")
        return
        
    dataset_cls = DATASET_REGISTRY[args.dataset]
    dataset = dataset_cls(root_dir=args.data_root) # Make sure your Dataset class accepts root_dir
    
    # 2. Chunking
    if args.max_samples:
        dataset.samples = dataset.samples[:args.max_samples]
    
    total = len(dataset)
    chunk_size = (total + args.num_chunks - 1) // args.num_chunks
    start_idx = args.chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total)
    
    # Slicing the internal list (assuming dataset.samples exists)
    if hasattr(dataset, 'samples') and isinstance(dataset.samples, list):
        dataset.samples = dataset.samples[start_idx:end_idx]
    
    print(f"   Processing indices {start_idx} -> {end_idx} (N={len(dataset)})")
    if len(dataset) == 0: return

    # 3. Load Model
    # Note: Ensure you have these wrappers in event_graph/models/
    if "Qwen" in args.backbone:
        from event_graph.models.qwen2_5_vl import Qwen2_5_VLWrapper
        # Use provided path -> env var -> HuggingFace ID
        path = args.model_path or os.environ.get("QWEN_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
        model = Qwen2_5_VLWrapper(model_path=path)
    elif "Video-LLaVA" in args.backbone:
        from event_graph.models.video_llava_7b import VideoLLaVAWrapper
        path = args.model_path or "LanguageBind/Video-LLaVA-7B-hf"
        model = VideoLLaVAWrapper(model_path=path)
    else:
        print(f"❌ Backbone {args.backbone} not implemented.")
        return

    # 4. Method
    processor = METHOD_REGISTRY[args.method](args, model)

    # 5. Loop
    results = []
    for sample in tqdm(dataset, desc=f"GPU {args.chunk_idx}"):
        try:
            if not sample.get('video_path'):
                raise FileNotFoundError("Video path is None")

            response = processor.process_and_inference(
                sample['video_path'],
                sample['question'],
                sample.get('options', [])
            )
            
            pred = extract_answer_from_text(response)
            results.append({
                "id": sample['id'],
                "pred": pred,
                "raw_response": response,
                "gt": sample.get('answer', ''),
                "q": sample.get('question', '')
            })
            
            # Save periodic
            if len(results) % 5 == 0:
                tmp = os.path.join(args.output_dir, f"tmp_{args.dataset}_{args.chunk_idx}.json")
                with open(tmp, 'w') as f: json.dump(results, f)

        except Exception as e:
            print(f"❌ Error ID {sample.get('id')}: {e}")
            results.append({"id": sample.get('id'), "error": str(e), "pred": "C"})

    # Final Save
    out_file = os.path.join(args.output_dir, f"{args.dataset}_{args.method}_chunk{args.chunk_idx}.json")
    with open(out_file, 'w') as f: json.dump(results, f, indent=2)
    print(f"✅ Saved to {out_file}")

if __name__ == "__main__":
    main()