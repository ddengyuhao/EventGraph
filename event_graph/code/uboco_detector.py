# /root/hhq/main_code/utils/uboco_detector.py
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class UbocoDetector:
    """
    Wrapper for Uboco: Unsupervised Boundary Contrastive Learning for Generic Event Boundary Detection.
    (Kang et al., CVPR 2022)
    
    该类实现了基于语义特征对比的边界检测。
    核心思想: 计算时序自相似度矩阵(Temporal Self-Similarity Matrix)，
    并在对角线上寻找差异最大的点作为事件边界。
    """
    def __init__(self, device='cuda'):
        self.device = device
        # 复用 CLIP 模型提取特征 (Uboco 原文使用 ResNet50，但在 Training-free 设定下，
        # 复用 LMM 的 Visual Encoder 是标准做法，且 CLIP 的语义性更强)
        self.model_name = "openai/clip-vit-large-patch14"
        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
        except:
            print("Warning: Could not load CLIP for Uboco, check internet connection.")

    def detect(self, video_path, sample_rate=1):
        """
        执行检测
        Args:
            sample_rate: 采样率，每几帧采一帧（为了速度）
        Returns:
            boundaries: List[float] (seconds)
        """
        # 1. 提取密集帧特征
        features, timestamps = self._extract_dense_features(video_path, sample_rate)
        
        if len(features) < 10:
            return []

        # 2. 计算余弦相似度变化 (Contrastive Score)
        # Uboco 原理: 边界处的 frame_t 与 frame_{t-1} 相似度低，
        # 且 frame_{t-window} 与 frame_{t+window} 相似度极低
        
        # 归一化特征
        features = features / features.norm(dim=-1, keepdim=True)
        
        # 计算相邻帧相似度 (N-1,)
        sim_scores = (features[:-1] * features[1:]).sum(dim=-1)
        
        # 转换为差异分数 (Dissimilarity)
        # 值越大，越可能是边界
        boundary_scores = 1.0 - sim_scores
        
        # 平滑分数 (Gaussian smoothing)
        boundary_scores = self._smooth_scores(boundary_scores.cpu().numpy())
        
        # 3. 峰值检测 (Peak Picking)
        # 动态阈值: 均值 + k * 标准差
        threshold = np.mean(boundary_scores) + 1.5 * np.std(boundary_scores)
        
        boundary_indices = []
        # 简单的局部最大值抑制 (NMS-like)
        min_dist = 5 # 最小间隔帧数
        
        for i in range(1, len(boundary_scores) - 1):
            is_peak = boundary_scores[i] > boundary_scores[i-1] and \
                      boundary_scores[i] > boundary_scores[i+1]
            
            if is_peak and boundary_scores[i] > threshold:
                # 检查距离
                if len(boundary_indices) == 0 or (i - boundary_indices[-1] > min_dist):
                    boundary_indices.append(i)
        
        # 4. 映射回时间戳
        # boundary_scores[i] 对应的是 timestamps[i] 和 timestamps[i+1] 之间
        detected_boundaries = []
        for idx in boundary_indices:
            t_boundary = (timestamps[idx] + timestamps[idx+1]) / 2.0
            detected_boundaries.append(t_boundary)
            
        return detected_boundaries

    def _extract_dense_features(self, video_path, sample_rate=4):
        """
        提取视频帧特征用于边界检测
        """
        frames = []
        timestamps = []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            
            frame_count += 1
            
        cap.release()
        
        # 批处理提取特征
        batch_size = 32
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i : i + batch_size]
                inputs = self.processor(images=batch_frames, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # 使用 image_embeds
                features = self.model.get_image_features(**inputs)
                all_features.append(features)
        
        if len(all_features) > 0:
            return torch.cat(all_features, dim=0), timestamps
        else:
            return torch.tensor([]), []

    def _smooth_scores(self, scores, window_size=5):
        """一维高斯平滑"""
        box = np.ones(window_size) / window_size
        scores_smooth = np.convolve(scores, box, mode='same')
        return scores_smooth