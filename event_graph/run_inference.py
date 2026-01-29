# /root/icml2026/event_graph/run_inference.py
import argparse
import os
import json
import torch
import re
from tqdm import tqdm
from my_dataset import DATASET_REGISTRY
from methods import METHOD_REGISTRY

def parse_args():
    parser = argparse.ArgumentParser(description="EventGraph-LLM Experiments")
    parser.add_argument("--dataset", type=str, default="CinePile", choices=["VideoMME", "CinePile", "VRBench", "LVBench"])
    parser.add_argument("--method", type=str, default="EventGraph-LMM")
    parser.add_argument("--backbone", type=str, default="Video-LLaVA-7B")
    parser.add_argument("--data_root", type=str, default="/root/icml2026/dataset")
    parser.add_argument("--token_budget", type=int, default=8192)
    parser.add_argument("--output_dir", type=str, default="./result")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)
    return parser.parse_args()

def extract_answer_from_text(text, options=None):
    """ Robust Answer Extraction """
    if not text: return "C"
    text = text.strip()
    
    # 1. Match "The answer is X"
    match = re.search(r'(?:answer|option)\s*(?:is|:)\s*[\(]?([A-D])[\)]?', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # 2. Match end of text " D." or "(D)"
    match = re.search(r'(?:^|\s)[\(]?([A-D])[\)]?[\.\s]*$', text)
    if match: return match.group(1).upper()

    # 3. Match start of text "D. The..." (Common in Qwen)
    match = re.search(r'^[\(]?([A-D])[\)]?[\.\s]', text)
    if match: return match.group(1).upper()

    return "C"

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Process {args.chunk_idx}/{args.num_chunks} starting on {device}")

    # 1. Load Dataset
    print(f"📂 Loading Dataset: {args.dataset}")
    try:
        dataset_cls = DATASET_REGISTRY[args.dataset]
        dataset = dataset_cls(root_dir=args.data_root)
    except KeyError:
        print(f"❌ Error: Dataset '{args.dataset}' not found in registry.")
        return

    # 2. Subset & Chunking
    if args.max_samples is not None:
        dataset.samples = dataset.samples[:args.max_samples]
        print(f"⚠️ DEBUG MODE: Truncated to first {args.max_samples} samples.")

    total_samples = len(dataset)
    chunk_size = (total_samples + args.num_chunks - 1) // args.num_chunks
    start_idx = args.chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_samples)
    
    dataset.samples = dataset.samples[start_idx:end_idx]
    print(f"🔄 Chunk {args.chunk_idx}: Processing samples {start_idx} to {end_idx} (Count: {len(dataset)})")

    if len(dataset) == 0:
        print("✅ Chunk is empty, exiting.")
        return

    # 3. Load Model
    print(f"🛠️ Loading Backbone: {args.backbone}")
    if args.backbone == "Video-LLaVA-7B":
        from models.video_llava_7b import VideoLLaVAWrapper
        model = VideoLLaVAWrapper()
    elif args.backbone == "Qwen2.5-VL-7B":
        try:
            from models.qwen2_5_vl import Qwen2_5_VLWrapper
            model = Qwen2_5_VLWrapper()
        except ImportError as e:
            print(f"❌ Failed to import Qwen Wrapper: {e}")
            return
    # 🔥 新增 72B 支持
    elif args.backbone == "Qwen2-VL-72B":
        from models.qwen2_vl_72b import Qwen2_VL_72B_Wrapper
        # 这里请填入你 72B 模型的真实绝对路径
        model = Qwen2_VL_72B_Wrapper(model_path="/root/hhq/models/Qwen/Qwen2___5-VL-72B-Instruct")

    else:
        print(f"❌ Unknown backbone: {args.backbone}")
        return

    # 4. Initialize Method
    method_cls = METHOD_REGISTRY[args.method]
    processor = method_cls(args, model)

    # 5. Inference
    results = []
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Start processing...")
    
    # 这里的 tqdm 迭代的是 dataset，没有 i 这个变量
    for sample in tqdm(dataset, desc=f"GPU {args.chunk_idx}"):
        try:
            # 检查视频是否存在
            if sample.get('video_path') is None:
                print(f"⚠️ Skipping ID {sample['id']} (Video missing)")
                results.append({
                    "id": sample['id'],
                    "pred": "C",
                    "gt": sample.get('answer', ''),
                    "q": sample.get('question', ''),
                    "error": "Video not found"
                })
                continue

            # 推理
            pred_raw = processor.process_and_inference(
                sample['video_path'],
                sample['question'],
                sample.get('options', [])
            )
            
            # 清洗答案
            pred_cleaned = extract_answer_from_text(pred_raw)
            
            results.append({
                "id": sample['id'],
                "pred": pred_cleaned,
                "pred_raw": pred_raw,
                "gt": sample.get('answer', ''),
                "q": sample.get('question', '')
            })
            
            # Save intermediate
            if len(results) % 5 == 0:
                temp_path = os.path.join(args.output_dir, f"temp_{args.dataset}_{args.chunk_idx}.json")
                with open(temp_path, 'w') as f: json.dump(results, f)

        except Exception as e:
            # 🔥 修复：使用 sample['id'] 而不是 i
            err_id = sample.get('id', 'unknown_id')
            print(f"❌ [Inference Error] ID: {err_id} | Error: {e}")
            # 记录错误以便后续分析
            results.append({
                "id": err_id,
                "error": str(e),
                "pred": "C",
                "gt": sample.get('answer', '')
            })

    # Final Save
    final_path = os.path.join(args.output_dir, f"{args.dataset}_{args.method}_chunk{args.chunk_idx}.json")
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✅ Saved to {final_path}")

if __name__ == "__main__":
    main()