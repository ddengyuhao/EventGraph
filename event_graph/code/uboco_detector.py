# /root/hhq/main_code/utils/uboco_detector.py
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class UbocoDetector:
    """
    Wrapper for Uboco: Unsupervised Boundary Contrastive Learning.
    Modified to accept external CLIP models for memory efficiency.
    """
    def __init__(self, device='cuda', clip_model=None, clip_processor=None):
        self.device = device
        
        # Optimization: Reuse existing CLIP model if provided
        if clip_model is not None and clip_processor is not None:
            self.model = clip_model
            self.processor = clip_processor
            print("  [Uboco] Reusing shared CLIP model.")
        else:
            self.model_name = "openai/clip-vit-large-patch14"
            try:
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
                self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Warning: Could not load CLIP for Uboco: {e}")

    def detect(self, video_path, sample_rate=1):
        # ... (Existing logic remains the same) ...
        # Ensure you include the rest of your original detect/extract logic here
        # ...
        return self._detect_internal(video_path, sample_rate)

    def _detect_internal(self, video_path, sample_rate):
        # Move your original detect() logic here to keep code clean
        features, timestamps = self._extract_dense_features(video_path, sample_rate)
        if len(features) < 10: return []
        
        features = features / features.norm(dim=-1, keepdim=True)
        sim_scores = (features[:-1] * features[1:]).sum(dim=-1)
        boundary_scores = 1.0 - sim_scores
        boundary_scores = self._smooth_scores(boundary_scores.cpu().numpy())
        
        threshold = np.mean(boundary_scores) + 1.5 * np.std(boundary_scores)
        boundary_indices = []
        min_dist = 5 
        
        for i in range(1, len(boundary_scores) - 1):
            is_peak = boundary_scores[i] > boundary_scores[i-1] and \
                      boundary_scores[i] > boundary_scores[i+1]
            if is_peak and boundary_scores[i] > threshold:
                if len(boundary_indices) == 0 or (i - boundary_indices[-1] > min_dist):
                    boundary_indices.append(i)
        
        detected_boundaries = []
        for idx in boundary_indices:
            t_boundary = (timestamps[idx] + timestamps[idx+1]) / 2.0
            detected_boundaries.append(t_boundary)
            
        return detected_boundaries

    def _extract_dense_features(self, video_path, sample_rate=4):
        # ... (Keep your original extraction logic) ...
        # Just ensure it uses self.model which is now potentially shared
        frames = []
        timestamps = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_count % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            frame_count += 1
        cap.release()
        
        # Batch processing (Optimization)
        batch_size = 64 # Increased batch size for A100
        all_features = []
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i : i + batch_size]
                inputs = self.processor(images=batch_frames, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                features = self.model.get_image_features(**inputs)
                all_features.append(features)
        
        if len(all_features) > 0:
            return torch.cat(all_features, dim=0), timestamps
        return torch.tensor([]), []

    def _smooth_scores(self, scores, window_size=5):
        box = np.ones(window_size) / window_size
        return np.convolve(scores, box, mode='same')