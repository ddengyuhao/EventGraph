import os 
import torch
import numpy as np
from PIL import Image
from .base_method import BaseMethod

try:
    from decord import VideoReader, cpu
except ImportError:
    print("⚠️ Warning: decord not installed")
    VideoReader = None

class BaselineUniform(BaseMethod):
    def __init__(self, args, model):
        super().__init__(args, model)
        # 这里的 token_budget 实际上变成了帧数控制
        # 建议先从 32 帧开始测，如果显存够大（A100 80G）可以试 64
        self.num_frames = 512
        print(f"📉 [Baseline] Uniform Sampling initialized with {self.num_frames} frames.")

    def _load_video_frames(self, video_path, num_frames):
        """均匀采样读取视频帧"""
        if not VideoReader:
            raise ImportError("decord is required for video loading.")
            
        if not os.path.exists(video_path):
            # 这里之前报错是因为没有 import os
            print(f"❌ Video not found: {video_path}") 
            return []

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # 均匀采样索引
            if total_frames <= num_frames:
                indices = np.arange(total_frames)
            else:
                indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            
            # 读取并转换为 PIL Image
            frames_np = vr.get_batch(indices).asnumpy()
            frames = [Image.fromarray(f) for f in frames_np]
            return frames
            
        except Exception as e:
            print(f"❌ Error loading video {video_path}: {e}")
            return []

    def _build_simple_prompt(self, question, options):
        """构建简单的 QA Prompt，不需要 Event Timeline"""
        # 格式化选项
        if isinstance(options, list) and options:
            # 清洗选项，确保都是字符串
            clean_opts = []
            for opt in options:
                clean_opts.append(str(opt))
                
            options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(clean_opts)])
            prompt = f"{question}\nOptions:\n{options_str}\nAnswer with the option letter directly."
        else:
            # 开放式问题
            prompt = f"{question}\nAnswer the question in detail."
            
        return prompt

    def process_and_inference(self, video_path, question, options):
        # 1. 直接均匀采样读取帧
        frames = self._load_video_frames(video_path, self.num_frames)
        
        if not frames:
            return "C" # 兜底

        # 2. 构建简单 Prompt
        prompt = self._build_simple_prompt(question, options)
        
        # 3. 直接调用模型生成
        # 注意：这里我们传入所有采样到的帧
        return self.model.generate(
            frames, 
            prompt, 
            options,
            max_new_tokens=10240 # Qwen 需要较长的输出空间
        )