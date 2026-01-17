import json
import glob
import os
import re

# === 配置 ===
# 你的结果文件夹路径
RESULT_DIR = "./results_VRBench"  
# 匹配文件名的模式
FILE_PATTERN = "VRBench_EventGraph-LMM_chunk*.json"

def clean_answer(text):
    """
    鲁棒的答案清洗函数：从长文本中提取 A/B/C/D
    """
    if not text: return "C" # 兜底
    text = str(text).strip()
    
    # 1. 已经是单个字母
    if len(text) == 1 and text.upper() in ['A', 'B', 'C', 'D']:
        return text.upper()
        
    # 2. 正则提取 "The answer is A" 或 "Answer: A"
    match = re.search(r'(?:answer|option)\s*(?:is|:)\s*[\(]?([A-D])[\)]?', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # 3. 提取文末的 "D." 或 "(D)"
    match = re.search(r'(?:^|\s)[\(]?([A-D])[\)]?[\.\s]*$', text)
    if match: return match.group(1).upper()
    
    # 4. 提取开头的 "D." (Qwen 常见)
    match = re.search(r'^[\(]?([A-D])[\)]?[\.\s]', text)
    if match: return match.group(1).upper()
    
    return text.strip()[0].upper() if text else "C"

def main():
    # 1. 寻找所有 chunk 文件
    search_path = os.path.join(RESULT_DIR, FILE_PATTERN)
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"❌ 未找到任何结果文件: {search_path}")
        return

    print(f"📂 找到 {len(files)} 个结果文件，开始合并...")
    
    all_results = []
    seen_ids = set()
    
    # 2. 合并数据
    for f_path in files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                
            print(f"   - {os.path.basename(f_path)}: 包含 {len(chunk_data)} 条数据")
            
            for item in chunk_data:
                # 去重 (防止某些 chunk 跑重了)
                if item['id'] not in seen_ids:
                    all_results.append(item)
                    seen_ids.add(item['id'])
                    
        except Exception as e:
            print(f"   ❌ 读取失败 {f_path}: {e}")

    total = len(all_results)
    if total == 0:
        print("❌ 没有有效数据。")
        return

    # 3. 计算准确率
    correct_count = 0
    missed_video_count = 0
    
    # 用于分析不同时长的准确率 (可选)
    # short_correct, short_total = 0, 0 ...
    
    print("\n🚀 开始评估...")
    
    for item in all_results:
        # 如果有 error 字段，说明视频下载失败，跳过或记错
        if "error" in item:
            missed_video_count += 1
            # 通常视作答错，或者根据需求剔除
            continue
            
        pred_raw = item.get('pred', '')
        # 优先用推理时清洗过的 pred，如果没有就现场清洗
        pred_final = clean_answer(pred_raw)
        
        gt = item.get('gt', '').strip().upper()
        
        if pred_final == gt:
            correct_count += 1
        
        # Debug: 打印几个错误案例看看
        # if pred_final != gt and total < 500: # 只在小样本时打印
        #    print(f"   [Wrong] ID: {item['id']} | Pred: {pred_final} (Raw: {pred_raw[:20]}...) | GT: {gt}")

    # 4. 输出报告
    accuracy = (correct_count / total) * 100
    if (total - missed_video_count) > 0:
        valid_acc = (correct_count / (total - missed_video_count)) * 100
    else:
        valid_acc = 0

    print("="*40)
    print(f"📊 Video-MME 最终评估报告")
    print("="*40)
    print(f"📥 总样本数 (Merged): {total}")
    print(f"⚠️ 视频缺失/失败数: {missed_video_count}")
    print(f"✅ 正确回答数: {correct_count}")
    print("-" * 20)
    print(f"🎯 总体准确率 (Overall Accuracy): {accuracy:.2f}%")
    if missed_video_count > 0:
        print(f"🎯 有效准确率 (Valid Accuracy):   {valid_acc:.2f}% (排除缺失视频)")
    print("="*40)
    
    # 5. 保存合并后的完整文件
    merged_path = os.path.join(RESULT_DIR, "VideoMME_FINAL_MERGED.json")
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"💾 合并后的完整结果已保存至: {merged_path}")

if __name__ == "__main__":
    main()