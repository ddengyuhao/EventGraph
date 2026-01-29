# # /root/hhq/main_code/datasets/longvideobench.py
# import os
# import json
# from .base_dataset import BaseDataset

# class LongVideoBenchDataset(BaseDataset):
#     def __init__(self, root_dir, duration_mode="all"):
#         super().__init__(root_dir)
        
#         # 路径配置
#         self.json_path = os.path.join(root_dir, "LongVideoBench", "lvb_val.json")
#         self.video_root = os.path.join(root_dir, "LongVideoBench", "videos")
        
#         # 如果文件不存在，仅打印警告（为了不阻断其他数据集运行）
#         if not os.path.exists(self.json_path):
#             print(f"Warning: LongVideoBench json not found at {self.json_path}")
#             return

#         print(f"[LongVideoBench] Loading from {self.json_path}...")
#         with open(self.json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
            
#         self.samples = []
#         for item in data:
#             # 这里的字段根据 LVB 实际 json 结构调整
#             # 假设 item 包含 video_id, question, candidates, correct_choice
#             video_id = item.get('video_id', '')
#             # 尝试拼接视频路径 (.mp4)
#             video_path = os.path.join(self.video_root, f"{video_id}.mp4")
            
#             # 只有当文件存在时才加入(可选)
#             # if not os.path.exists(video_path): continue

#             self.samples.append({
#                 "id": item.get('id', video_id),
#                 "video_path": video_path,
#                 "question": item.get('question', ''),
#                 "options": item.get('candidates', []), # 选项列表
#                 "answer": item.get('correct_choice', 'C'), # A/B/C/D
#                 "duration": item.get('duration', 0)
#             })
            
#         print(f"[LongVideoBench] Loaded {len(self.samples)} samples.")

import os
import json
import glob
from torch.utils.data import Dataset

class LongVideoBenchDataset(Dataset):
    def __init__(self, root_dir="/root/icml2026/dataset/LongVideoBench", split="test"):
        self.root_dir = root_dir
        self.samples = []
        
        print(f"📂 [LongVideoBench] 初始化数据集，根目录: {self.root_dir}")

        # 1. 寻找元数据文件 (JSON)
        json_files = glob.glob(os.path.join(self.root_dir, "LongVideoBench", "lvb_val.json"))
        if not json_files:
            # 尝试去上一级找，或者常见命名
            json_files = glob.glob(os.path.join(self.root_dir, "..", "*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"❌ 未找到 .json 元数据文件！请检查路径: {self.root_dir}")
            
        json_path = json_files[0]
        print(f"📄 加载元数据: {os.path.basename(json_path)}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
        except Exception as e:
            print(f"❌ JSON 加载失败: {e}")
            return

        print(f"   共加载了 {len(data_list)} 条题目。")

        # 2. 建立本地视频索引
        # 假设视频在 videos/ 子目录下，或者直接在根目录
        print("🔍 扫描本地视频文件...")
        video_map = {}
        # 递归搜索 mp4 和 mkv
        all_videos = sorted(glob.glob(os.path.join(self.root_dir, "**", "*.mp4"), recursive=True)) + \
                     sorted(glob.glob(os.path.join(self.root_dir, "**", "*.mkv"), recursive=True))
        
        for v_path in all_videos:
            fname = os.path.basename(v_path)
            fid = os.path.splitext(fname)[0]
            video_map[fname] = v_path # 完整文件名匹配 (如 86CxyhFV9MI.mp4)
            video_map[fid] = v_path   # ID 匹配 (如 86CxyhFV9MI)

        print(f"   硬盘上实际找到 {len(all_videos)} 个视频文件。")

        # 3. 构建样本
        skipped_count = 0
        for entry in data_list:
            # LongVideoBench 的 JSON 结构:
            # "video_path": "86CxyhFV9MI.mp4"
            # "video_id": "86CxyhFV9MI"
            vid_filename = entry.get('video_path', '')
            vid_id = entry.get('video_id', '')

            # 尝试匹配视频
            video_path = None
            if vid_filename in video_map:
                video_path = video_map[vid_filename]
            elif vid_id in video_map:
                video_path = video_map[vid_id]
            
            if video_path is None:
                skipped_count += 1
                continue

            # 处理选项 (candidates -> options)
            candidates = entry.get('candidates', [])
            
            # 处理答案 (correct_choice index -> A/B/C/D)
            correct_idx = entry.get('correct_choice')
            if correct_idx is not None and isinstance(correct_idx, int):
                answer_letter = chr(65 + correct_idx) # 0->A, 1->B
            else:
                answer_letter = "C" # 兜底

            self.samples.append({
                "id": entry.get('id', f"{vid_id}_{correct_idx}"),
                "video_path": video_path,
                "question": entry.get('question', ''),
                "options": candidates, # 这里的 candidates 就是选项列表
                "answer": answer_letter
            })

        print(f"✅ 数据集构建完成！")
        print(f"   - 跳过缺失视频: {skipped_count}")
        print(f"   - 有效样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]