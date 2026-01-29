import os
import json
import glob
from torch.utils.data import Dataset

class VRBenchDataset(Dataset):
    def __init__(self, root_dir="/root/icml2026/dataset/VRBench", split="test"):
        # 1. 路径修正
        if os.path.basename(root_dir.rstrip("/")) == "videos":
            self.root_dir = os.path.dirname(root_dir.rstrip("/"))
            print(f"🔄 [VRBench] 路径修正: 从 videos 回退到 {self.root_dir}")
            
        # 情况B: 传入的是 .../dataset (父目录) -> 自动进入 VRBench 子目录
        elif os.path.exists(os.path.join(root_dir, "VRBench")):
            self.root_dir = os.path.join(root_dir, "VRBench")
            print(f"🔄 [VRBench] 路径修正: 自动进入子目录 {self.root_dir}")
            
        else:
            self.root_dir = root_dir
            
        self.samples = []
        print(f"📂 [VRBench] 数据集根目录: {self.root_dir}")

        # === 2. 寻找元数据文件 ===
        # 优先找 VRBench_eval.json 或 .jsonl
        candidates = [
            os.path.join(self.root_dir, "*.json"),
            os.path.join(self.root_dir, "*.jsonl"),
            # 防御性编程：也在上一级找找
            os.path.join(os.path.dirname(self.root_dir), "*.json"),
        ]
        
        json_path = None
        for pattern in candidates:
            files = glob.glob(pattern)
            if files:
                json_path = sorted(files)[0] # 排序取第一个，保证确定性
                break
        
        if not json_path:
            print(f"❌ [Error] 在 {self.root_dir} 未找到元数据文件(.json/.jsonl)。")
            print(f"   当前目录下有: {os.listdir(self.root_dir) if os.path.exists(self.root_dir) else '目录不存在'}")
            raise FileNotFoundError("无法找到元数据文件，请检查是否已下载 VRBench_eval.json")
            
        print(f"📄 加载元数据: {os.path.basename(json_path)}")

        # 3. 加载 JSON 数据
        data_list = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                if json_path.endswith('.jsonl'):
                    for line in f:
                        if line.strip(): data_list.append(json.loads(line))
                else:
                    content = json.load(f)
                    if isinstance(content, list):
                        data_list = content
                    elif isinstance(content, dict):
                        data_list = content.get('videos', []) or content.get('data', [])
        except Exception as e:
            print(f"❌ JSON 解析失败: {e}")
            return

        # 4. 建立本地视频索引 (这是关键!)
        print("🔍 扫描本地已解压的视频文件...")
        video_map = {}
        all_videos = sorted(glob.glob(os.path.join(self.root_dir, "**", "*.mp4"), recursive=True)) + \
                     sorted(glob.glob(os.path.join(self.root_dir, "**", "*.avi"), recursive=True))
        
        for v_path in all_videos:
            fname = os.path.basename(v_path) # e.g., "0wEsr-o4yHo.mp4"
            fid = os.path.splitext(fname)[0] # e.g., "0wEsr-o4yHo"
            
            # 建立多重映射，保证能被找到
            video_map[fname] = v_path 
            video_map[fid] = v_path 

        print(f"   硬盘上实际找到 {len(all_videos)} 个视频。")
        if len(all_videos) == 0:
            print("⚠️ 警告: 你似乎没有解压任何视频，或者视频不在 videos/ 子目录下。")

        # 5. 过滤并构建样本 (只保留有视频的题目)
        skipped_count = 0
        for entry in data_list:
            # 获取各种可能的 ID
            vid_id = entry.get('video_id') or entry.get('video_uid')
            # 有些 JSON 的 video_path 字段里包含文件名
            json_vpath = entry.get('video_path', '') 
            json_fname = os.path.basename(json_vpath) if json_vpath else ""

            # 尝试匹配本地文件
            actual_path = None
            
            # 1. 尝试用 ID 匹配
            if vid_id and vid_id in video_map:
                actual_path = video_map[vid_id]
            # 2. 尝试用 ID + .mp4 匹配
            elif vid_id and f"{vid_id}.mp4" in video_map:
                actual_path = video_map[f"{vid_id}.mp4"]
            # 3. 尝试用 JSON 里的文件名匹配
            elif json_fname and json_fname in video_map:
                actual_path = video_map[json_fname]
            
            # === 核心修改: 如果找不到视频，直接跳过 ===
            if actual_path is None:
                skipped_count += 1
                continue 

            # 解析 MCQ 问题
            mcqs = entry.get('mcq', {})
            if not mcqs: continue

            for qa_key, qa_data in mcqs.items():
                # 提取选项
                raw_options = qa_data.get('options', {})
                options_list = []
                for k in sorted(raw_options.keys()):
                    options_list.append(raw_options[k])
                
                self.samples.append({
                    "id": f"{vid_id}_{qa_key}",
                    "video_path": actual_path, # 这里的路径一定存在
                    "question": qa_data.get('question', ''),
                    "options": options_list,
                    "answer": qa_data.get('answer', '')
                })

        print(f"✅ 数据集构建完成！")
        print(f"   - 跳过缺失视频的条目: {skipped_count}")
        print(f"   - 有效测试样本数: {len(self.samples)} (仅包含本地存在的视频)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]