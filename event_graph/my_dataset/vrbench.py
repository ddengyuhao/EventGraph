import os
import json
import glob
from torch.utils.data import Dataset

class VRBenchDataset(Dataset):
    def __init__(self, root_dir="/root/ICML2026/dataset/VRBench", split="test"):
        self.root_dir = root_dir
        self.samples = []
        
        # -------------------------------------------------------
        # 根据 VRBench 的实际文件结构修改下面的加载逻辑
        # 假设结构是：root_dir/videos/*.mp4 和 root_dir/prompts.json
        # -------------------------------------------------------
        
        # 1. 查找所有视频文件
        video_dir = os.path.join(root_dir, "videos") # 假设视频在这个子目录
        if not os.path.exists(video_dir):
             # 容错：如果视频就在根目录
             video_dir = root_dir
             
        # 这里的后缀可能需要根据实际情况添加 .avi, .mkv 等
        video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        
        print(f"Found {len(video_files)} videos in {video_dir}")

        # 2. 加载对应的 Prompt/Question
        # VRBench 往往是视频生成或检索 benchmark，可能没有标准的 QA 选项 (A,B,C,D)
        # 既然你要用 EventGraph 跑，我们把它构造成一个 "描述题" 或者 "检索匹配题"
        
        for idx, video_path in enumerate(video_files):
            filename = os.path.basename(video_path)
            
            # 构造样本
            self.samples.append({
                "id": f"vrbench_{idx}",
                "video_path": video_path,
                # 如果没有专门的 questions 文件，就构造一个通用的问题
                "question": "Describe the events and details in this video in detail.", 
                "options": [], # 开放式问题没有选项
                "answer": ""   # 测试集通常没有答案，或者是 caption
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]