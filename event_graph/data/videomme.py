import os
import json
from torch.utils.data import Dataset

class VideoMMEDataset(Dataset):
    """
    Dataset loader for Video-MME.
    Expected structure:
        root_dir/
            test.json (Annotation file)
            videos/   (Directory containing video files)
    """
    def __init__(self, root_dir="./dataset/VideoMME", split="test", **kwargs):
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, "videos")
        annotation_path = os.path.join(root_dir, "test.json") # Adjust based on actual file name

        if not os.path.exists(annotation_path):
            print(f"⚠️ Warning: Annotation file not found at {annotation_path}")
            self.samples = []
        else:
            with open(annotation_path, 'r') as f:
                self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Adapt keys based on actual VideoMME json structure
        video_id = item.get('video_id', f"v_{idx}")
        video_filename = f"{video_id}.mp4" 
        video_path = os.path.join(self.video_dir, video_filename)
        
        if not os.path.exists(video_path):
            video_path = None # Handle missing video gracefully

        return {
            "id": video_id,
            "video_path": video_path,
            "question": item['question'],
            "options": item['options'],
            "answer": item['answer']
        }