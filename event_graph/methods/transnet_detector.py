import torch
import numpy as np
from pathlib import Path

try:
    from transnetv2_pytorch import TransNetV2
    TRANSNET_AVAILABLE = True
except ImportError:
    TRANSNET_AVAILABLE = False
    print("⚠️  TransNet V2 not available. Install with: pip install transnetv2-pytorch")


class TransNetV2Detector:
    """
    TransNet V2 Shot Boundary Detector
    """
    
    def __init__(self, device='cuda'):
        """        
        Args:
            device: 'cuda' 或 'cpu'
        """
        if not TRANSNET_AVAILABLE:
            raise ImportError(
                "TransNet V2 is required for shot detection. "
                "Install with: pip install transnetv2-pytorch"
            )
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print("[TransNet V2] Loading pretrained model...")
        self.model = TransNetV2()
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"  ✓ TransNet V2 loaded on {self.device}")
    
    def detect_shots(self, video_path, threshold=0.5):
"""
        Detect shot boundaries in a video.
        
        Args:
            video_path (str): Path to the video file.
            threshold (float): Detection threshold (default 0.5, as per paper).
        
        Returns:
            list: List of (start_time, end_time) tuples representing N variable-length segments.
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"[TransNet V2] Detecting shots in: {Path(video_path).name}")
        
        try:
            from decord import VideoReader, cpu
        except ImportError:
            raise ImportError("decord is required. Install with: pip install decord")
        
        vr = VideoReader(str(video_path), ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        print(f"  Video info: {total_frames} frames, {fps:.2f} fps")
        
        batch_size = 500
        all_predictions = []
        
        print(f"  Processing video in chunks of {batch_size} frames...")
        
        for start_idx in range(0, total_frames, batch_size):
            end_idx = min(start_idx + batch_size, total_frames)
            chunk_size = end_idx - start_idx
            
            indices = list(range(start_idx, end_idx))
            frames_chunk = vr.get_batch(indices).asnumpy()  # (chunk_size, H, W, 3)
            
            from PIL import Image
            resized_frames = []
            for frame in frames_chunk:
                pil_img = Image.fromarray(frame)
                pil_img = pil_img.resize((48, 27), Image.BILINEAR)  # W x H
                resized_frames.append(np.array(pil_img))
            
            video_tensor = np.stack(resized_frames, axis=0)  # (chunk_size, 27, 48, 3)
            video_tensor = np.expand_dims(video_tensor, axis=0)  # (1, chunk_size, 27, 48, 3)
            video_tensor = torch.from_numpy(video_tensor).to(torch.uint8)
            video_tensor = video_tensor.to(self.device)
            
            with torch.no_grad():
                predictions_chunk = self.model(video_tensor)[0]  # (predictions, indices)
            
            if torch.is_tensor(predictions_chunk):
                predictions_chunk = predictions_chunk.cpu().numpy()
            
            predictions_chunk = np.squeeze(predictions_chunk)
            
            all_predictions.append(predictions_chunk)
            
            del video_tensor
            torch.cuda.empty_cache()
            
            if end_idx % 1000 == 0 or end_idx == total_frames:
                print(f"    Processed {end_idx}/{total_frames} frames")
        
        predictions = np.concatenate(all_predictions, axis=0)
        
        if len(predictions.shape) > 1:
            predictions = predictions.squeeze()
        
        print(f"  Prediction stats: min={predictions.min():.3f}, max={predictions.max():.3f}, mean={predictions.mean():.3f}")
        
        boundary_frames = self._find_boundaries_with_nms(predictions, threshold, min_distance=15)
        
        events = self._boundaries_to_events(
            boundary_frames,
            total_frames=total_frames,
            fps=fps
        )
        
        print(f"  ✓ Detected {len(events)} shots")
        return events
    
    def _find_boundaries_with_nms(self, predictions, threshold=0.5, min_distance=15):
        candidates = np.where(predictions > threshold)[0]
        
        if len(candidates) == 0:
            return np.array([], dtype=np.int64)
        
        boundaries = []
        i = 0
        while i < len(candidates):
            current_frame = candidates[i]
            current_score = predictions[current_frame]
            
            j = i
            while j < len(candidates) and candidates[j] - current_frame < min_distance:
                if predictions[candidates[j]] > current_score:
                    current_frame = candidates[j]
                    current_score = predictions[current_frame]
                j += 1
            
            boundaries.append(current_frame)
            
            i = j
        
        boundary_frames = np.array(boundaries, dtype=np.int64)
        print(f"  NMS: {len(candidates)} candidates → {len(boundary_frames)} boundaries")
        
        return boundary_frames
    
    def _boundaries_to_events(self, boundary_frames, total_frames, fps):
        """
        Returns:
            events: list of (start_time, end_time)
        """
        if len(boundary_frames) == 0:
            duration = total_frames / fps
            return [(0.0, duration)]
        
        events = []
        
        if boundary_frames[0] > 0:
            start_time = 0.0
            end_time = boundary_frames[0] / fps
            events.append((start_time, end_time))
        
        for i in range(len(boundary_frames) - 1):
            start_time = boundary_frames[i] / fps
            end_time = boundary_frames[i + 1] / fps
            events.append((start_time, end_time))
        
        if boundary_frames[-1] < total_frames - 1:
            start_time = boundary_frames[-1] / fps
            end_time = total_frames / fps
            events.append((start_time, end_time))
        
        return events
    
    def _get_fps(self, video_path):
        """        
        Args:
            video_path
        
        Returns:
            fps
        """
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(str(video_path), ctx=cpu(0))
            fps = vr.get_avg_fps()
            return fps
        except Exception as e:
            print(f"  ⚠️  Could not get FPS, defaulting to 30: {e}")
            return 30.0  
    
    def merge_short_segments(self, events, min_duration=0.5):
        """
        Args:
            events: list of (start, end)
            min_duration
        Returns:
            merged_events
        """
        if len(events) == 0:
            return events
        
        merged = []
        current_start, current_end = events[0]
        
        for start, end in events[1:]:
            duration = current_end - current_start
            
            if duration < min_duration:
                current_end = end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        
        return merged


# Fallback: PySceneDetect
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    PYSCENEDETECT_AVAILABLE = True
except ImportError:
    PYSCENEDETECT_AVAILABLE = False


class PySceneDetectFallback:
    def __init__(self):
        if not PYSCENEDETECT_AVAILABLE:
            raise ImportError("PySceneDetect not available")
        print("[Fallback] Using PySceneDetect instead of TransNet V2")
    
    def detect_shots(self, video_path, threshold=30.0):
        video_manager = VideoManager([str(video_path)])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        
        scene_list = scene_manager.get_scene_list()
        
        events = []
        for scene in scene_list:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            events.append((start_time, end_time))
        
        video_manager.release()
        
        print(f"  ✓ Detected {len(events)} shots (PySceneDetect)")
        return events

