import argparse
import os
import json
import torch
import re
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset

# 引入 EventGraph 核心逻辑
from methods.eventgraph import EventGraphLMM

# === 1. 模型工厂函数 (Model Factory) ===
def load_model_wrapper(backbone_name):
    """
    根据 backbone 名称动态加载对应的模型 Wrapper
    """
    print(f"🛠️ 正在加载模型 Wrapper: {backbone_name} ...")
    
    if backbone_name == "Video-LLaVA-7B":
        from models.video_llava_7b import VideoLLaVAWrapper
        return VideoLLaVAWrapper()
        
    elif backbone_name == "Qwen2.5-VL-7B":
        # 确保你之前创建了 models/qwen2_5_vl.py
        try:
            from models.qwen2_5_vl import Qwen2_5_VLWrapper
            return Qwen2_5_VLWrapper()
        except ImportError as e:
            print(f"❌ 无法导入 Qwen Wrapper: {e}")
            print("请检查 models/qwen2_5_vl.py 是否存在且依赖已安装。")
            exit(1)
            
    elif "34B" in backbone_name:
        try:
            from models.llava_next_34b import LLaVANext34BWrapper
            return LLaVANext34BWrapper()
        except ImportError:
            print("❌ 无法导入 LLaVA-NeXT-34B Wrapper")
            exit(1)
            
    else:
        raise ValueError(f"未知的 Backbone: {backbone_name}")

# === 2. 智能映射 Dataset (保持不变) ===
class CinePileSmartDataset(Dataset):
    def __init__(self, root_dir, max_samples=50):
        self.video_dir = os.path.join(root_dir, "yt_videos")
        print(f"📂 加载 CinePile (Smart Mode), 视频目录: {self.video_dir}")
        
        # 加载元数据
        hf_dataset = load_dataset("tomg-group-umd/cinepile", split="test")
        self.hf_dataset = hf_dataset.select(range(max_samples)) # 只取前50个
        
        self.ans_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        link = row['yt_clip_link']
        
        # 智能映射: 根据链接找到 v1.mp4 或 v2.mp4
        if "duU5cdQtpSE" in link:
            video_filename = "v1.mp4"
        elif "VDwI61e2_6I" in link:
            video_filename = "v2.mp4"
        else:
            video_filename = "unknown.mp4"

        video_path = os.path.join(self.video_dir, video_filename)
        
        actual_path = video_path if os.path.exists(video_path) else None
            
        return {
            "id": f"cinepile_{idx}",
            "video_path": actual_path,
            "question": row['question'],
            "options": row['choices'],
            "answer": self.ans_map.get(row['answer_key_position'], "C")
        }

# === 3. 增强版答案清洗函数 ===
def clean_prediction(pred_text):
    """
    针对 Qwen 等 Chat 模型可能输出的一句话进行清洗，提取选项。
    """
    if not pred_text: return "C"
    
    # 1. 最简单的：如果第一个字符就是 A-E
    first_char = pred_text.strip()[0].upper()
    if first_char in ['A', 'B', 'C', 'D', 'E']:
        return first_char
        
    # 2. 正则匹配 "Answer: A" 或 "The answer is (A)"
    # 匹配模式：单词边界 + (Answer|Option) + 非字母字符 + (A-E)
    match = re.search(r'(?:Answer|Option|is)\s*[:\-\s]*([A-E])\b', pred_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # 3. 匹配括号 "(A)"
    match = re.search(r'\(([A-E])\)', pred_text)
    if match:
        return match.group(1).upper()
        
    # 4. 如果都失败了，但在文本里出现了某个选项加点 "A."
    for opt in ['A', 'B', 'C', 'D', 'E']:
        if f"{opt}." in pred_text:
            return opt
            
    return "C" # 兜底

# === 4. 主评测逻辑 ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/root/icml2026/dataset/CinePile")
    parser.add_argument("--output_file", type=str, default="result_top50_smart.json")
    parser.add_argument("--token_budget", type=int, default=2048)
    # ✨ 新增：支持命令行指定 backbone
    parser.add_argument("--backbone", type=str, default="Qwen2.5-VL-7B", 
                        choices=["Video-LLaVA-7B", "Qwen2.5-VL-7B", "LLaVA-NeXT-Video-34B"])
    args = parser.parse_args()
    
    args.method = "EventGraph-LMM"
    
    # 1. 动态加载模型
    print(f"🚀 初始化 Backbone: {args.backbone} ...")
    model = load_model_wrapper(args.backbone)
    
    # 2. 初始化方法 (EventGraph 会根据 args.backbone 调整 token 估算逻辑)
    print("🧠 初始化 EventGraph 插件...")
    processor = EventGraphLMM(args, model)

    # 3. 加载数据
    print("📚 加载数据集 (Smart Mapping)...")
    dataset = CinePileSmartDataset(root_dir=args.data_root, max_samples=50)

    results = []
    correct_count = 0
    valid_count = 0

    print(f"▶️ 开始推理 (Dataset: CinePile Top50 | Model: {args.backbone})...")
    
    for sample in tqdm(dataset):
        if sample['video_path'] is None:
            results.append({"id": sample['id'], "pred": "C", "gt": sample['answer'], "valid": False})
            continue

        try:
            # 核心推理
            pred_raw = processor.process_and_inference(
                sample['video_path'],
                sample['question'],
                sample.get('options', [])
            )
            
            # 清洗答案
            pred_cleaned = clean_prediction(pred_raw)
            
            # 统计
            gt = sample['answer']
            is_correct = (pred_cleaned == gt)
            
            if is_correct:
                correct_count += 1
            valid_count += 1
            
            # 打印简报 (为了防止刷屏，可以把 raw pred 截断)
            raw_show = (pred_raw[:20] + '..') if len(pred_raw) > 20 else pred_raw
            icon = '✅' if is_correct else '❌'
            tqdm.write(f"  {sample['id']} | Pred: {pred_cleaned} (Raw: {raw_show}) | GT: {gt} | {icon}")
            
            results.append({
                "id": sample['id'],
                "pred": pred_cleaned,
                "pred_raw": pred_raw, # 保存原始输出以便后续分析
                "gt": gt,
                "is_correct": is_correct,
                "valid": True
            })
            
        except Exception as e:
            tqdm.write(f"❌ Error {sample['id']}: {e}")

    # === 5. 最终效果评估 ===
    if valid_count > 0:
        accuracy = (correct_count / valid_count) * 100
        print("\n" + "="*40)
        print(f"📊 实验报告 (Model: {args.backbone})")
        print(f"📥 有效样本数: {valid_count}")
        print(f"✅ 正确回答数: {correct_count}")
        print(f"🎯 准确率 (Accuracy): {accuracy:.2f}%")
        print("="*40)
        
        # 保存文件名带上模型名字
        final_output = args.output_file.replace(".json", f"_{args.backbone}.json")
        with open(final_output, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"📄 结果已保存至: {final_output}")
    else:
        print("❌ 没有有效数据被测试。")

if __name__ == "__main__":
    main()