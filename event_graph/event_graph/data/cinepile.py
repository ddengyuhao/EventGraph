import os
import subprocess
from datasets import load_dataset
from torch.utils.data import Dataset

class CinePileDataset(Dataset):
    """
    Dataset loader for CinePile.
    Automatically downloads videos from YouTube using yt-dlp if not present locally.
    """
    def __init__(self, root_dir="./dataset/CinePile", split="test", **kwargs):
        """
        Args:
            root_dir (str): Root directory to save/load videos.
            split (str): Dataset split ('train', 'test').
        """
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, "yt_videos")
        os.makedirs(self.video_dir, exist_ok=True)
        
        print(f"📂 Loading CinePile [{split}] metadata from HuggingFace...")
        try:
            self.hf_dataset = load_dataset("tomg-group-umd/cinepile", split=split)
        except Exception as e:
            print(f"❌ Failed to load CinePile metadata: {e}")
            self.hf_dataset = []

        self.samples = list(range(len(self.hf_dataset)))
        self.ans_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        self.failed_urls = set()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        
        # Sanitize filename
        safe_title = "".join([c if c.isalnum() else "_" for c in row['yt_clip_title']])
        video_filename = f"{row['movie_name']}_{safe_title}.mp4"
        video_path = os.path.join(self.video_dir, video_filename)
        
        question = row['question']
        options = row['choices']
        answer = self.ans_map.get(row['answer_key_position'], "C")

        # Check blacklist to avoid repeated failed requests
        if row['yt_clip_link'] in self.failed_urls:
            return self._build_sample(idx, None, question, options, answer)
        
        # Download if missing
        if not os.path.exists(video_path):
            print(f"⬇️ Downloading: {row['yt_clip_title']}")
            success = self._download_video(row['yt_clip_link'], video_path)
            if not success:
                self.failed_urls.add(row['yt_clip_link'])
                return self._build_sample(idx, None, question, options, answer)

        return self._build_sample(idx, video_path, question, options, answer)

    def _build_sample(self, idx, video_path, question, options, answer):
        return {
            "id": f"cinepile_{idx}",
            "video_path": video_path,
            "question": question,
            "options": options,
            "answer": answer
        }

    def _download_video(self, url, output_path):
        """
        Downloads video using yt-dlp.
        Configuration:
        - Respects system HTTP_PROXY/HTTPS_PROXY.
        - Uses 'cookies.txt' from root_dir if it exists.
        """
        cookies_path = os.path.join(self.root_dir, "cookies.txt")
        
        cmd = [
            "yt-dlp",
            "-S", "height:224,ext:mp4:m4a",
            "--recode", "mp4",
            "--no-check-certificate",
            "-o", output_path,
            url
        ]

        # Optional: Add cookies if present in dataset dir
        if os.path.exists(cookies_path):
            cmd.extend(["--cookies", cookies_path])
        
        # Optional: Add Proxy from env
        # (Users should set export HTTP_PROXY=http://... in their shell)

        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                print(f"⚠️ [yt-dlp Error] {url}: {result.stderr.strip()[:200]}...")
                return False
            return True
        except Exception as e:
            print(f"❌ Download Exception: {e}")
            return False