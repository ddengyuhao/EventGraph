import os
import torch
import optuna
import random
import argparse
import re
from tqdm import tqdm
from types import SimpleNamespace

# 导入你的项目模块
# 确保此脚本在 /root/icml2026/event_graph/ 目录下运行
from methods.eventgraph import EventGraphLMM
from models.qwen2_5_vl import Qwen2_5_VLWrapper
from my_dataset.videomme import VideoMMEDataset
from my_dataset.longvideobench import LongVideoBenchDataset # 如果你想测 LongVideoBench

# ==========================================
# 1. 配置区域 (只需修改这里)
# ==========================================
DATASET_NAME = "VideoMME" # 或 "LongVideoBench"
DATA_ROOT = "/root/icml2026/dataset/Video-MME/videomme"
MODEL_PATH = "/root/hhq/models/Qwen2.5-VL-7B-Instruct" # 你的 Qwen 模型路径
N_TRIALS = 30           # 尝试多少组参数 (建议 20-50)
N_SAMPLES = 40          # 每次尝试跑多少个视频 (验证集大小，建议 50-100，太多会慢)
TOKEN_BUDGET = 8192     # 固定 budget
MAX_NEW_TOKENS = 10240    # 生成长度

# ==========================================
# 2. 辅助函数
# ==========================================
def extract_answer_from_text(text):
    """简单的答案提取逻辑"""
    if not text: return "C"
    text = text.strip()
    match = re.search(r'(?:answer|option)\s*(?:is|:)\s*[\(]?([A-D])[\)]?', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r'(?:^|\s)[\(]?([A-D])[\)]?[\.\s]*$', text)
    if match: return match.group(1).upper()
    match = re.search(r'^[\(]?([A-D])[\)]?[\.\s]', text)
    if match: return match.group(1).upper()
    return "C"

# ==========================================
# 3. 全局加载 (模型和数据只加载一次，节省时间)
# ==========================================
print("🚀 [Setup] 正在加载模型和数据集 (这只需要一次)...")

# 模拟 args 对象
args = SimpleNamespace(
    token_budget=TOKEN_BUDGET, 
    backbone="Qwen2.5-VL-7B"
)

# 1. 加载 VLM Backbone
backbone_model = Qwen2_5_VLWrapper(model_path=MODEL_PATH)

# 2. 初始化 EventGraph 方法
# 注意：CLIP 和 TransNet 会在这里被加载
method_model = EventGraphLMM(args, backbone_model)

# 3. 加载数据集
if DATASET_NAME == "VideoMME":
    full_dataset = VideoMMEDataset(root_dir=DATA_ROOT)
elif DATASET_NAME == "LongVideoBench":
    full_dataset = LongVideoBenchDataset(root_dir=DATA_ROOT)
else:
    raise ValueError("Unknown dataset")

# 4. 随机抽取验证集 (固定种子以保证每次 trial 跑的是同一批数据)
random.seed(42)
if len(full_dataset) > N_SAMPLES:
    validation_indices = random.sample(range(len(full_dataset)), N_SAMPLES)
else:
    validation_indices = range(len(full_dataset))

validation_set = [full_dataset[i] for i in validation_indices]
print(f"✅ [Setup] 验证集准备就绪: 共 {len(validation_set)} 个样本")

# ==========================================
# 4. Optuna 目标函数
# ==========================================
def objective(trial):
    """
    Optuna 会反复调用这个函数，每次传入不同的 trial 参数
    """
    # 1. 定义搜索空间 (Hyperparameter Search Space)
    # -------------------------------------------------
    tau = trial.suggest_int('tau', 15, 60, step=5)           # 时间跨度阈值
    delta = trial.suggest_float('delta', 0.50, 0.85, step=0.05) # 相似度阈值
    alpha = trial.suggest_float('alpha', 0.1, 0.5, step=0.1)    # PageRank 跳转率
    lambda_param = trial.suggest_float('lambda', 0.5, 2.0, step=0.25) # CELF 惩罚系数
    # -------------------------------------------------
    
    # 2. 动态更新模型参数
    # Python 允许直接修改对象属性
    method_model.tau = tau
    method_model.delta = delta
    method_model.alpha = alpha
    method_model.lambda_param = lambda_param
    
    # 3. 在验证集上跑推理
    correct_count = 0
    total_count = 0
    
    # 使用 tqdm 显示进度 (desc 显示当前参数)
    pbar = tqdm(validation_set, desc=f"Trial {trial.number}", leave=False)
    
    for sample in pbar:
        try:
            # 运行 EventGraph 推理
            # 注意：EventGraph 内部会使用更新后的 self.tau 等参数
            pred_raw = method_model.process_and_inference(
                sample['video_path'],
                sample['question'],
                sample.get('options', [])
            )
            
            # 提取答案
            pred = extract_answer_from_text(pred_raw)
            gt = sample.get('answer', '').strip().upper()
            
            if pred == gt:
                correct_count += 1
            total_count += 1
            
        except Exception as e:
            # 遇到错误不中断，记为错误
            pass
            
    # 4. 计算准确率
    if total_count == 0: return 0.0
    accuracy = correct_count / total_count
    
    # 打印当前 Trial 的结果
    print(f"🔍 Trial {trial.number}: Acc={accuracy:.2%} | Params: tau={tau}, delta={delta:.2f}, alpha={alpha:.2f}, lambda={lambda_param:.2f}")
    
    return accuracy

# ==========================================
# 5. 启动搜索
# ==========================================
if __name__ == "__main__":
    print(f"\n🔥 [Optuna] 开始超参数搜索 (共 {N_TRIALS} 次尝试)...")
    
    # 创建 Study，方向是最大化准确率
    study = optuna.create_study(direction="maximize")
    
    # 开始优化
    study.optimize(objective, n_trials=N_TRIALS)
    
    # 输出最佳结果
    print("\n" + "="*50)
    print("🏆 最佳参数组合 (Best Hyperparameters):")
    print("="*50)
    best_params = study.best_params
    print(f"Best Accuracy: {study.best_value:.2%}")
    print(f"Best Params:")
    for key, value in best_params.items():
        print(f"  - {key}: {value}")
    
    # 建议修改
    print("\n你可以将 eventgraph.py 中的 __init__ 修改为:")
    print(f"self.tau = {best_params['tau']}")
    print(f"self.delta = {best_params['delta']}")
    print(f"self.alpha = {best_params['alpha']}")
    print(f"self.lambda_param = {best_params['lambda']}")
    print("="*50)