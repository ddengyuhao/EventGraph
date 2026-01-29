import os
import json
from torch.utils.data import Dataset

class VRBenchDataset(Dataset):
    def __init__(self, root_dir="./dataset/VRBench", split="test", **kwargs):
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, "videos")
        # Assuming standard JSONL or JSON annotation
        self.samples = []
        
        # Mock loading logic - replace with actual VRBench logic
        # For a formal project, ensure this matches the official VRBench format
        ann_file = os.path.join(root_dir, "VRBench.json")
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                self.samples = json.load(f)
        else:
            print(f"⚠️ VRBench annotation not found at {ann_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        video_path = os.path.join(self.video_dir, item.get('video_name', ''))
        
        return {
            "id": item.get('id', idx),
            "video_path": video_path if os.path.exists(video_path) else None,
            "question": item.get('question', ''),
            "options": item.get('options', []),
            "answer": item.get('answer', 'C')
        }