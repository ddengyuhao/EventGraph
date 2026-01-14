# datasets/cinepile.py
import os
import subprocess
from datasets import load_dataset
from torch.utils.data import Dataset

class CinePileDataset(Dataset):
    def __init__(self, root_dir="/root/hhq/dataset/CinePile", split="test", duration_mode=None):
        """
        Args:
            root_dir: 视频保存和查找的根目录
            split: 数据集划分 ('train', 'test')
        """
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, "yt_videos")
        os.makedirs(self.video_dir, exist_ok=True)
        
        print(f"Loading CinePile [{split}] dataset from HuggingFace...")
        # 加载 HF 数据集
        self.hf_dataset = load_dataset("tomg-group-umd/cinepile", split=split)
        
        # 为了兼容 run_inference.py 的逻辑，我们需要一个 samples 列表
        # 这里我们做一个轻量级的映射，不预先下载所有视频
        self.samples = list(range(len(self.hf_dataset)))
        
        # 答案映射: 0->A, 1->B, ...
        self.ans_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 获取原始数据
        row = self.hf_dataset[idx]
        
        # 构造视频文件名 (使用 movie_name + clip_id 以防重名)
        # 注意：你需要确保文件名清洗以避免非法字符
        safe_title = "".join([c if c.isalnum() else "_" for c in row['yt_clip_title']])
        video_filename = f"{row['movie_name']}_{safe_title}.mp4"
        video_path = os.path.join(self.video_dir, video_filename)
        
        # 1. 检查视频是否存在，不存在则尝试下载
        if not os.path.exists(video_path):
            print(f"[CinePile] Downloading: {row['yt_clip_title']}...")
            success = self._download_video(row['yt_clip_link'], video_path)
            if not success:
                raise FileNotFoundError(f"Failed to download video: {row['yt_clip_link']}")

        # 2. 构造 EventGraph 所需的格式
        # 格式: {'video_path': str, 'question': str, 'options': List[str], 'answer': str, 'id': str}
        return {
            "id": f"cinepile_{idx}",
            "video_path": video_path,
            "question": row['question'],
            "options": row['choices'], # List of strings
            "answer": self.ans_map.get(row['answer_key_position'], "C") 
        }

    def _download_video(self, url, output_path):
        """集成你提供的 yt-dlp 下载逻辑"""
        try:
            # 临时文件名，避免下载中断导致文件损坏
            cmd = [
                "yt-dlp",
                "-S", "height:224,ext:mp4:m4a",
                "--recode", "mp4",
                "-o", output_path,
                url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Download Error: {e}")
            return False