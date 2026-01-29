# # /root/hhq/main_code/datasets/videomme.py
# import os
# import json
# import re
# from .base_dataset import BaseDataset

# class VideoMMEDataset(BaseDataset):
#     def __init__(self, root_dir, duration_mode="all"):
#         super().__init__(root_dir)
        
#         self.json_path = os.path.join(root_dir, "Video-MME", "video_mme.json")
#         self.video_root = os.path.join(root_dir, "Video-MME", "videos") 
        
#         print(f"[VideoMME] Loading json from {self.json_path}...")
#         with open(self.json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
            
#         # 1. 建立视频索引 (这步必须保留，否则找不到文件)
#         self.video_map = self._build_video_index(self.video_root)
        
#         self.samples = []
#         video_count = 0
        
#         print("🚀 [VideoMME] FORCE LOAD MODE: Loading EVERYTHING found on disk...")
        
#         for item in data:
#             # 2. 查找视频路径
#             video_id = str(item.get('video_id', '')).strip()
#             url = item.get('url', '')
            
#             fpath = self.video_map.get(video_id)
#             if not fpath and url:
#                 yt_id = self._extract_youtube_id(url)
#                 if yt_id: fpath = self.video_map.get(yt_id)
#             if not fpath and video_id.startswith("_"):
#                 fpath = self.video_map.get(video_id[1:])

#             # 如果找不到文件，物理上没法跑，跳过
#             if not fpath: 
#                 continue
            
#             video_count += 1

#             # 3. 构造样本 (不做任何筛选)
#             # 给一个默认时长防止除零报错
#             dummy_duration = 3600 

#             for q in item['questions']:
#                 self.samples.append({
#                     "id": f"{video_id}_{q['question_id']}",
#                     "video_path": fpath,
#                     "duration": dummy_duration,
#                     "question": q['question'],
#                     "options": q['options'],
#                     "answer": q['answer'],
#                     "task_type": q['task_type']
#                 })
        
#         print(f"✅ Successfully loaded {len(self.samples)} samples from {video_count} videos.")

#         # === 冒烟测试强制截断已禁用 ===
#         # 现在由 run_inference.py 的 --max_samples 参数控制样本数
#         # ============================

#     def _build_video_index(self, root_dir):
#         print(f"    -> Indexing videos in {root_dir}...")
#         idx = {}
#         for root, _, files in os.walk(root_dir):
#             for file in files:
#                 if file.lower().endswith(('.mp4', '.mkv', '.webm', '.avi')):
#                     name = os.path.splitext(file)[0]
#                     full_path = os.path.join(root, file)
#                     idx[name] = full_path
#                     if name.startswith("_"):
#                         idx[name[1:]] = full_path
#         print(f"    -> Indexed {len(idx)} files.")
#         return idx

#     def _extract_youtube_id(self, url):
#         if not isinstance(url, str): return None
#         patterns = [r"v=([a-zA-Z0-9_-]{11})", r"youtu\.be/([a-zA-Z0-9_-]{11})"]
#         for p in patterns:
#             match = re.search(p, url)
#             if match: return match.group(1)
#         return None

import os
import glob
import pandas as pd
from torch.utils.data import Dataset

class VideoMMEDataset(Dataset):
    def __init__(self, root_dir="/root/icml2026/dataset/Video-MME/videomme", split="test"):
        self.root_dir = root_dir
        # 1. 智能路径探测
        potential_subdirs = ["Video-MME", "videomme", ".cache"]
        for sub in potential_subdirs:
            sub_path = os.path.join(self.root_dir, sub)
            if os.path.exists(sub_path):
                self.root_dir = sub_path
        
        print(f"📂 [Video-MME] 数据集根目录: {self.root_dir}")

        # 2. 加载元数据 (.parquet)
        parquet_files = glob.glob(os.path.join(self.root_dir, "**", "*.parquet"), recursive=True)
        if not parquet_files:
            parquet_files = glob.glob(os.path.join(os.path.dirname(self.root_dir), "**", "*.parquet"), recursive=True)
            
        if not parquet_files:
            raise FileNotFoundError(f"❌ 未找到 .parquet 文件！搜索范围: {self.root_dir}")
        
        parquet_path = parquet_files[0]
        print(f"📄 加载元数据: {os.path.basename(parquet_path)}")

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"❌ Parquet 读取失败: {e}")
            return

        print(f"   元数据包含 {len(df)} 条记录。")
        
        # 3. 扫描视频文件
        search_roots = [
            self.root_dir, 
            os.path.join(self.root_dir, "videos"),
            os.path.dirname(parquet_path)
        ]
        
        video_map = {}
        all_videos = []
        for search_root in set(search_roots):
            if os.path.exists(search_root):
                found = sorted(glob.glob(os.path.join(search_root, "**", "*.mp4"), recursive=True)) + \
                        sorted(glob.glob(os.path.join(search_root, "**", "*.mkv"), recursive=True))
                all_videos.extend(found)
        
        all_videos = list(set(all_videos)) # 去重

        for v_path in all_videos:
            fname = os.path.basename(v_path)       # "1uqupftxFOM.mp4"
            fid = os.path.splitext(fname)[0]       # "1uqupftxFOM"
            video_map[fname] = v_path
            video_map[fid] = v_path
            
        print(f"🔍 硬盘上实际找到 {len(all_videos)} 个视频文件。")

        # 4. 构建样本 (核心修复: URL ID 提取)
        self.samples = []
        skipped_count = 0
        
        def extract_youtube_id(url):
            if not isinstance(url, str): return None
            # 处理标准格式 https://www.youtube.com/watch?v=ID
            if "v=" in url:
                return url.split("v=")[1].split("&")[0]
            # 处理短链 https://youtu.be/ID
            elif "youtu.be/" in url:
                return url.split("youtu.be/")[1].split("?")[0]
            return None

        for index, row in df.iterrows():
            # 策略1: 从 URL 提取 (最可靠)
            candidates = []
            if 'url' in row:
                yt_id = extract_youtube_id(row['url'])
                if yt_id: candidates.append(yt_id)
            
            # 策略2: 尝试 videoID 列 (有些数据集这个列存的是真实ID)
            if 'videoID' in row:
                candidates.append(str(row['videoID']).strip())
                
            # 策略3: 原始 video_id (虽然看起来是序号 '001'，但也试一下)
            candidates.append(str(row['video_id']).strip())
            
            # 逐个尝试匹配
            video_path = None
            for key in candidates:
                if key in video_map:
                    video_path = video_map[key]
                    break
                elif f"{key}.mp4" in video_map:
                    video_path = video_map[f"{key}.mp4"]
                    break
            
            if video_path is None:
                skipped_count += 1
                continue

            options = row['options']
            if hasattr(options, 'tolist'):
                options = options.tolist()
            
            self.samples.append({
                "id": f"vmme_{candidates[0]}_{index}", # 使用找到的第一个ID作为key
                "video_path": video_path,
                "question": row['question'],
                "options": options,
                "answer": row['answer']
            })

        print(f"✅ 数据集构建完成！")
        print(f"   - 跳过缺失视频: {skipped_count}")
        print(f"   - 有效样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# import os
# import glob
# import pandas as pd
# from torch.utils.data import Dataset

# class VideoMMEDataset(Dataset):
#     def __init__(self, root_dir="/root/icml2026/dataset/Video-MME/videomme", split="test"):
#         # === 1. 智能路径探测 ===
#         # 即使传入的是通用的 dataset 根目录，也能自动找到 Video-MME
#         self.root_dir = root_dir
#         print(f"📂 [Video-MME] 初始化，搜索根目录: {self.root_dir}")

#         # 优先检查常见的子目录名，缩小搜索范围
#         potential_subdirs = ["Video-MME", "videomme", ".cache"]
#         for sub in potential_subdirs:
#             sub_path = os.path.join(self.root_dir, sub)
#             if os.path.exists(sub_path):
#                 self.root_dir = sub_path
#                 print(f"   -> 自动进入子目录: {self.root_dir}")

#         # === 2. 深度搜索元数据 (.parquet) ===
#         # 使用 recursive=True 穿透所有子文件夹 (.cache, videomme 等)
#         print("🔍 正在深度搜索 .parquet 元数据文件...")
#         parquet_files = glob.glob(os.path.join(self.root_dir, "**", "*.parquet"), recursive=True)
        
#         if not parquet_files:
#             # 如果没找到，尝试回退到上一级再搜一次（防御性编程）
#             parent_dir = os.path.dirname(self.root_dir)
#             parquet_files = glob.glob(os.path.join(parent_dir, "**", "*.parquet"), recursive=True)

#         if not parquet_files:
#             print(f"❌ 错误: 未找到 .parquet 文件。")
#             print(f"   搜索路径包括: {self.root_dir} 及其所有子目录")
#             raise FileNotFoundError("无法找到 Video-MME 的元数据文件")
        
#         # 通常只有一个 parquet，取第一个
#         parquet_path = parquet_files[0]
#         print(f"📄 锁定元数据: {parquet_path}")

#         try:
#             df = pd.read_parquet(parquet_path)
#         except Exception as e:
#             print(f"❌ Parquet 读取失败 (需要 pip install pandas pyarrow): {e}")
#             return

#         print(f"   元数据包含 {len(df)} 条记录。")

#         # === 3. 深度搜索视频文件 ===
#         # 根据截图，视频可能在 videos/data/ 下，所以必须递归搜索
#         print(f"🔍 正在深度搜索视频文件 (.mp4/.mkv)...")
#         # 这里的 root_dir 已经被更新为包含 parquet 的目录，通常视频也在附近
#         search_roots = [
#             self.root_dir, 
#             os.path.join(self.root_dir, "videos"),
#             os.path.dirname(parquet_path) # parquet 所在目录
#         ]
        
#         video_map = {}
#         all_videos = []
        
#         for search_root in set(search_roots): # 去重
#             if os.path.exists(search_root):
#                 found = sorted(glob.glob(os.path.join(search_root, "**", "*.mp4"), recursive=True)) + \
#                         sorted(glob.glob(os.path.join(search_root, "**", "*.mkv"), recursive=True))
#                 all_videos.extend(found)
        
#         # 去重（因为可能多次搜到同一个文件）
#         all_videos = list(set(all_videos))

#         for v_path in all_videos:
#             fname = os.path.basename(v_path)
#             fid = os.path.splitext(fname)[0] 
#             video_map[fname] = v_path
#             video_map[fid] = v_path
            
#             # Video-MME 特例处理：有时候 ID 不包含后缀，但文件名乱七八糟
#             # 如果你的 ID 是 "0ag_Qi5OEd0"，文件名也是 "0ag_Qi5OEd0.mp4"，上面的 fid 就能匹配

#         print(f"   硬盘上实际找到 {len(all_videos)} 个视频文件。")
#         if len(all_videos) == 0:
#             print(f"⚠️ 警告: 未找到视频！请确认已解压到 {self.root_dir} 下的某个子目录")

#         # === 4. 构建样本 ===
#         self.samples = []
#         skipped_count = 0
        
#         for index, row in df.iterrows():
#             vid_id = str(row['video_id'])
            
#             # 匹配逻辑
#             video_path = None
#             if vid_id in video_map:
#                 video_path = video_map[vid_id]
#             elif f"{vid_id}.mp4" in video_map:
#                 video_path = video_map[f"{vid_id}.mp4"]
            
#             # 如果没找到视频（因为可能只解压了一部分），跳过
#             if video_path is None:
#                 skipped_count += 1
#                 continue

#             options = row['options']
#             if hasattr(options, 'tolist'):
#                 options = options.tolist()
            
#             self.samples.append({
#                 "id": f"vmme_{vid_id}_{index}",
#                 "video_path": video_path,
#                 "question": row['question'],
#                 "options": options,
#                 "answer": row['answer']
#             })

#         print(f"✅ 数据集构建完成！")
#         print(f"   - 跳过缺失视频: {skipped_count}")
#         print(f"   - 有效样本数: {len(self.samples)}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]