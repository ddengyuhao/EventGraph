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

        self.failed_urls = set()

    def __len__(self):
        return len(self.samples)

   # /root/icml2026/event_graph/my_dataset/cinepile.py

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        
        # 构造视频文件名
        safe_title = "".join([c if c.isalnum() else "_" for c in row['yt_clip_title']])
        video_filename = f"{row['movie_name']}_{safe_title}.mp4"
        video_path = os.path.join(self.video_dir, video_filename)

        # --- 新增：黑名单检查逻辑 ---
        # 如果这个链接之前失败过，直接跳过，防止反复请求
        if row['yt_clip_link'] in self.failed_urls:
            print(f"⏩ Skipping known failed video: {row['yt_clip_link']}")
            return {
                "id": f"cinepile_{idx}",
                "video_path": None,  # <--- 标记为空，而不是写 ...
                "question": row['question'],
                "options": row['choices'],
                "answer": self.ans_map.get(row['answer_key_position'], "C") 
            }
        # ---------------------------
        
        # 正常的下载逻辑
        if not os.path.exists(video_path):
            success = self._download_video(row['yt_clip_link'], video_path)
            if not success:
                print(f"⚠️ Adding {row['yt_clip_link']} to blacklist.")
                self.failed_urls.add(row['yt_clip_link'])
                # 下载失败也返回 None
                return {
                    "id": f"cinepile_{idx}",
                    "video_path": None, # <--- 标记为空
                    "question": row['question'],
                    "options": row['choices'],
                    "answer": self.ans_map.get(row['answer_key_position'], "C") 
                }

        # 正常返回
        return {
            "id": f"cinepile_{idx}",
            "video_path": video_path,
            "question": row['question'],
            "options": row['choices'], # List of strings
            "answer": self.ans_map.get(row['answer_key_position'], "C") 
        }

    # datasets/cinepile.py (局部修改)

    def _download_video(self, url, output_path):
        """集成 yt-dlp 下载逻辑，显式指定代理"""
        # 定义你的代理地址
        proxy_url = "http://agent.baidu.com:8891"
        # 你的 cookies 路径
        cookies_path = "/root/ICML2026/cookies.txt" 
        
        try:
            cmd = [
                "yt-dlp",
                "--proxy", proxy_url,
                "--cookies", cookies_path,  # <---【核心修改】加入 Cookies
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "-S", "height:224,ext:mp4:m4a",
                "--recode", "mp4",
                "--no-check-certificate",
                "-o", output_path,
                url
            ]
            
            # 使用 check=True 可以在下载失败时抛出异常，方便你捕获
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=False # 先设为False，手动判断returncode
            )
            
            if result.returncode != 0:
                # 打印详细错误日志，方便你看为什么下载失败
                print(f"⚠️ [yt-dlp Error] {url}: {result.stderr.strip()}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Download Exception: {e}")
            return False