# # /root/icml2026/event_graph/code/eventgraph.py
# import torch
# import numpy as np
# import cv2  # <--- 修复报错: 之前缺少这个导入
# import os
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# from .base_method import BaseMethod
# from .graph_builder import compute_similarity_matrix, compute_pagerank_matrix
# from .celf_solver import CELFSelector
# from .uboco_detector import UbocoDetector

# try:
#     from decord import VideoReader, cpu
# except ImportError:
#     print("⚠️ Warning: decord not installed")
#     VideoReader = None

# class EventGraphLMM(BaseMethod):
#     def __init__(self, args, model):
#         super().__init__(args, model)
        
#         # Params from Paper Section 4.1
#         self.tau = 30.0  
#         self.delta = 0.65 
#         self.alpha = 0.15 
#         self.lambda_param = 1.0 
#         self.token_budget = args.token_budget
        
#         # Detect token density for budget calculation
#         backbone_name = getattr(args, 'backbone', '')
#         if '34B' in backbone_name:
#             self.tokens_per_frame = 576 
#         elif 'Qwen' in backbone_name:
#             self.tokens_per_frame = 256 
#         else:
#             self.tokens_per_frame = 256 # Video-LLaVA-7B default
            
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # 1. Load CLIP Once
#         self._load_clip_model()
        
#         # 2. Initialize Uboco with the SHARED model
#         self.shot_detector = UbocoDetector(
#             device=self.device, 
#             clip_model=self.clip_model, 
#             clip_processor=self.clip_processor
#         )

#     def _load_clip_model(self):
#         # Uses local path if available to save download time
#         local_path = "/root/hhq/models/clip-vit-large-patch14"
#         model_name = local_path if os.path.exists(local_path) else "openai/clip-vit-large-patch14"
        
#         try:
#             self.clip_processor = CLIPProcessor.from_pretrained(model_name)
#             self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
#             self.clip_model.eval()
#         except Exception as e:
#             print(f"Warning: Loading CLIP from {model_name} failed ({e}), trying openai default.")
#             self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
#             self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
#             self.clip_model.eval()

#     def _detect_shot_boundaries(self, video_path):
#         # Delegate to the shared-model Uboco detector
#         try:
#             # Uboco return timestamps (e.g., [2.5, 5.1, ...])
#             boundaries = self.shot_detector.detect(video_path, sample_rate=2) 
#             # Convert boundaries to event intervals (start, end)
#             events = self._boundaries_to_events(boundaries, video_path)
            
#             # Filter noise < 1s
#             events = [e for e in events if (e[1] - e[0]) >= 1.0]
            
#             # If too few events, use fallback
#             if len(events) < 3: 
#                 return self._fallback_windows(video_path)
#             return events
#         except Exception as e:
#             print(f"  [EventGraph] Uboco failed ({e}), using fallback.")
#             return self._fallback_windows(video_path)

#     def _boundaries_to_events(self, boundaries, video_path):
#         """
#         修复报错: 将时间戳列表转换为 (start, end) 元组列表
#         """
#         # 获取视频总时长
#         cap = cv2.VideoCapture(video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         duration = frame_count / fps if fps > 0 else 0
#         cap.release()

#         if duration == 0: return []

#         # 构造区间
#         sorted_bounds = sorted([0.0] + list(boundaries) + [duration])
#         # 去重
#         sorted_bounds = sorted(list(set(sorted_bounds)))
        
#         events = []
#         for i in range(len(sorted_bounds) - 1):
#             start = sorted_bounds[i]
#             end = sorted_bounds[i+1]
#             events.append((start, end))
#         return events

#     def _extract_event_features(self, video_path, events):
#         # Batch processing for CLIP
#         if VideoReader is None: raise ImportError("decord required")
#         vr = VideoReader(video_path, ctx=cpu(0))
#         fps = vr.get_avg_fps()
        
#         representative_frames = []
#         valid_indices = []
        
#         for idx, (start_t, end_t) in enumerate(events):
#             mid_t = (start_t + end_t) / 2.0
#             # 安全检查防止越界
#             if mid_t * fps >= len(vr): continue
            
#             frame_idx = min(len(vr) - 1, int(mid_t * fps))
#             frame_np = vr[frame_idx].asnumpy()
#             representative_frames.append(Image.fromarray(frame_np))
#             valid_indices.append(idx)
        
#         # Batch Process
#         batch_size = 32
#         global_feats_list = []
#         local_feats_list = []
        
#         with torch.no_grad():
#             for i in range(0, len(representative_frames), batch_size):
#                 batch = representative_frames[i : i+batch_size]
#                 inputs = self.clip_processor(images=batch, return_tensors="pt", padding=True)
#                 inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
#                 # Global (CLS)
#                 g_feats = self.clip_model.get_image_features(**inputs)
#                 global_feats_list.append(g_feats)
                
#                 # Local (Patches)
#                 outputs = self.clip_model.vision_model(**inputs, output_hidden_states=True)
#                 l_feats = outputs.last_hidden_state[:, 1:, :] # Remove CLS
#                 local_feats_list.append(l_feats)
        
#         if len(global_feats_list) == 0:
#             return torch.tensor([]), torch.tensor([]), []

#         global_feats = torch.cat(global_feats_list, dim=0)
#         local_feats = torch.cat(local_feats_list, dim=0)
        
#         return global_feats, local_feats, representative_frames

#     def _construct_event_graph(self, global_feats, local_feats, events):
#         """
#         构建语义-时序图
#         """
#         N = global_feats.shape[0]
#         # 1. Compute Semantic Adjacency (Eq. 3, 4)
#         adj_semantic = compute_similarity_matrix(
#             global_feats, local_feats, 
#             tau=self.tau, 
#             event_times=events, 
#             threshold=self.delta
#         )
#         # 2. Compute Reachability (PageRank, Eq. 6)
#         Pi = compute_pagerank_matrix(adj_semantic, alpha=self.alpha)
#         return Pi

#     def _select_subgraph(self, Pi, question, global_feats, events):
#         """
#         CELF 算法选择关键子图
#         """
#         # 1. Encode Question
#         inputs = self.clip_processor(text=[question], return_tensors="pt", padding=True)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         with torch.no_grad():
#             q_feat = self.clip_model.get_text_features(**inputs)
#             q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
        
#         # 2. Calculate Query Relevance (Eq. 5)
#         g_norm = global_feats / global_feats.norm(dim=-1, keepdim=True)
#         relevance = torch.mm(g_norm, q_feat.t()).squeeze() # (N,)
#         relevance = torch.clamp(relevance, min=0.0) # ReLU
        
#         # 3. Calculate Cost (Token consumption)
#         # 简单的线性代价: 每个事件消耗 tokens_per_frame
#         costs = torch.full((len(events),), self.tokens_per_frame, device=self.device)
        
#         # 4. CELF Selection
#         selector = CELFSelector(Pi, relevance, costs, lambda_param=self.lambda_param)
#         selected_indices = selector.select(budget=self.token_budget)
        
#         return selected_indices

#     def _fallback_windows(self, video_path):
#         """
#         如果 Uboco 失败，使用等间隔切片
#         """
#         cap = cv2.VideoCapture(video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         cap.release()
        
#         duration = count / fps if fps > 0 else 0
#         events = []
#         # 每2秒切一段
#         step = 2.0
#         for t in np.arange(0, duration, step):
#             events.append((t, min(t + step, duration)))
        
#         # 如果视频极短或者读不到，给一个默认
#         if not events:
#             events = [(0.0, 1.0)]
            
#         return events

#     def _build_graph_cot_prompt(self, question, options, segments, adj_matrix, selected_indices):
#         event_timeline = [f"Event{i+1}" for i, _, _ in segments]
        
#         # 格式化选项字符串
#         if isinstance(options, list):
#             options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
#         else:
#             options_str = str(options)

#         prompt = (
#             f"Question: {question}\n"
#             f"Options:\n{options_str}\n"
#             f"Key Events Timeline: {' -> '.join(event_timeline)}\n"
#             f"Based on these visual events, reason step-by-step and choose the best answer."
#         )
#         return prompt

#     def process_and_inference(self, video_path, question, options):
#         # 1. 检测事件
#         events = self._detect_shot_boundaries(video_path)
#         if not events: return "C"
        
#         # 2. 提取特征
#         global_feats, local_feats, frames = self._extract_event_features(video_path, events)
#         if len(frames) == 0: return "C"

#         # 3. 建图
#         Pi = self._construct_event_graph(global_feats, local_feats, events)
        
#         # 4. 选图
#         sel_idx = self._select_subgraph(Pi, question, global_feats, events)
#         if not sel_idx: sel_idx = [0] # Fallback
        
#         # 5. 准备推理数据
#         selected_frames = [frames[i] for i in sorted(sel_idx)]
#         selected_segments = [(events[i][0], events[i][1], i) for i in sorted(sel_idx)]
        
#         # 6. 生成 Prompt
#         prompt = self._build_graph_cot_prompt(
#             question, options, 
#             selected_segments, 
#             Pi, sel_idx
#         )
        
#         # 7. 调用 VLM 推理
#         return self.model.generate(selected_frames, prompt, options)


# /root/icml2026/event_graph/code/eventgraph.py
import torch
import numpy as np
import cv2
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from .base_method import BaseMethod
from .graph_builder import compute_similarity_matrix, compute_pagerank_matrix
from .celf_solver import CELFSelector

# === 修改 1: 导入 TransNetV2Detector ===
# 假设你把代码保存为了 transnet_detector.py
try:
    from .transnet_detector import TransNetV2Detector
except ImportError:
    print("⚠️ Warning: Could not import TransNetV2Detector. Make sure transnet_detector.py exists.")
    TransNetV2Detector = None

try:
    from decord import VideoReader, cpu
except ImportError:
    print("⚠️ Warning: decord not installed")
    VideoReader = None

class EventGraphLMM(BaseMethod):
    def __init__(self, args, model):
        super().__init__(args, model)
        
        # Params from Paper Section 4.1
        self.tau = 30.0  
        self.delta = 0.65 
        self.alpha = 0.15 
        self.lambda_param = 1.0 
        self.token_budget = args.token_budget
        
        # Detect token density for budget calculation
        backbone_name = getattr(args, 'backbone', '')
        if '34B' in backbone_name:
            self.tokens_per_frame = 576 
        elif 'Qwen' in backbone_name:
            # Qwen是动态分辨率，这里给一个经验平均值
            self.tokens_per_frame = 512 
        else:
            self.tokens_per_frame = 256 # Video-LLaVA-7B default
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load CLIP Once (用于后续特征提取和建图，不用于检测了)
        self._load_clip_model()
        
        # === 修改 2: 初始化 TransNet V2 ===
        if TransNetV2Detector is not None:
            print("🚀 [EventGraph] Initializing TransNet V2 Detector...")
            self.shot_detector = TransNetV2Detector(device='cuda') # 强制使用cuda如果可用
        else:
            self.shot_detector = None
            print("❌ [EventGraph] TransNet V2 Detector not available. Will use fallback.")

    def _load_clip_model(self):
        # Uses local path if available to save download time
        local_path = "/root/hhq/models/clip-vit-large-patch14"
        model_name = local_path if os.path.exists(local_path) else "openai/clip-vit-large-patch14"
        
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.clip_model.eval()
        except Exception as e:
            print(f"Warning: Loading CLIP from {model_name} failed ({e}), trying openai default.")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.clip_model.eval()

    def _detect_shot_boundaries(self, video_path):
        """
        使用 TransNet V2 进行镜头分割
        """
        # 如果检测器没初始化，直接回退
        if self.shot_detector is None:
            return self._fallback_windows(video_path)

        try:
            # === 修改 3: 直接调用 detect_shots ===
            # TransNetV2Detector.detect_shots 已经返回了 [(start, end), ...] 格式
            # 不需要再手动转换 boundaries_to_events
            events = self.shot_detector.detect_shots(video_path, threshold=0.5)
            
            # 过滤极短的噪声片段 (< 0.5s)
            events = [e for e in events if (e[1] - e[0]) >= 0.5]
            
            # 如果检测到的事件太少，说明可能是长镜头或者检测失败，使用回退策略
            if len(events) < 1: 
                print(f"  [EventGraph] Too few shots detected ({len(events)}), using fallback.")
                return self._fallback_windows(video_path)
            
            return events

        except Exception as e:
            print(f"  ❌ [EventGraph] TransNet V2 failed ({e}), using fallback.")
            return self._fallback_windows(video_path)

    # _boundaries_to_events 函数现在可以删除了，因为 TransNet 类内部处理了
    # 但为了防止某些子类继承调用，你可以保留它，或者直接删除以保持代码整洁。
    # 这里我把它移除了。

    def _extract_event_features(self, video_path, events):
        # Batch processing for CLIP
        if VideoReader is None: raise ImportError("decord required")
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        
        representative_frames = []
        valid_indices = []
        
        for idx, (start_t, end_t) in enumerate(events):
            mid_t = (start_t + end_t) / 2.0
            # 安全检查防止越界
            if mid_t * fps >= len(vr): 
                # 尝试取最后一张
                frame_idx = len(vr) - 1
            else:
                frame_idx = min(len(vr) - 1, int(mid_t * fps))
            
            try:
                frame_np = vr[frame_idx].asnumpy()
                representative_frames.append(Image.fromarray(frame_np))
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error extracting frame at {mid_t}s: {e}")
                continue
        
        if not representative_frames:
            return torch.tensor([]), torch.tensor([]), []

        # Batch Process
        batch_size = 32
        global_feats_list = []
        local_feats_list = []
        
        with torch.no_grad():
            for i in range(0, len(representative_frames), batch_size):
                batch = representative_frames[i : i+batch_size]
                inputs = self.clip_processor(images=batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Global (CLS)
                g_feats = self.clip_model.get_image_features(**inputs)
                global_feats_list.append(g_feats)
                
                # Local (Patches)
                outputs = self.clip_model.vision_model(**inputs, output_hidden_states=True)
                l_feats = outputs.last_hidden_state[:, 1:, :] # Remove CLS
                local_feats_list.append(l_feats)
        
        if len(global_feats_list) == 0:
            return torch.tensor([]), torch.tensor([]), []

        global_feats = torch.cat(global_feats_list, dim=0)
        local_feats = torch.cat(local_feats_list, dim=0)
        
        return global_feats, local_feats, representative_frames

    def _construct_event_graph(self, global_feats, local_feats, events):
        """
        构建语义-时序图
        """
        # 确保数据在同一设备
        global_feats = global_feats.to(self.device)
        local_feats = local_feats.to(self.device)
        
        N = global_feats.shape[0]
        # 1. Compute Semantic Adjacency (Eq. 3, 4)
        adj_semantic = compute_similarity_matrix(
            global_feats, local_feats, 
            tau=self.tau, 
            event_times=events, 
            threshold=self.delta
        )
        # 2. Compute Reachability (PageRank, Eq. 6)
        Pi = compute_pagerank_matrix(adj_semantic, alpha=self.alpha)
        return Pi

    def _select_subgraph(self, Pi, question, global_feats, events):
        """
        CELF 算法选择关键子图
        """
        # 1. Encode Question
        inputs = self.clip_processor(text=[question], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            q_feat = self.clip_model.get_text_features(**inputs)
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
        
        # 2. Calculate Query Relevance (Eq. 5)
        # Normalize global feats
        g_norm = global_feats / global_feats.norm(dim=-1, keepdim=True)
        relevance = torch.mm(g_norm, q_feat.t()).squeeze() # (N,)
        
        # Handle shape mismatch if only 1 event
        if relevance.dim() == 0:
            relevance = relevance.unsqueeze(0)
            
        relevance = torch.clamp(relevance, min=0.0) # ReLU
        
        # 3. Calculate Cost (Token consumption)
        # 简单的线性代价: 每个事件消耗 tokens_per_frame
        costs = torch.full((len(events),), self.tokens_per_frame, device=self.device)
        
        # 4. CELF Selection
        selector = CELFSelector(Pi, relevance, costs, lambda_param=self.lambda_param)
        selected_indices = selector.select(budget=self.token_budget)
        
        return selected_indices

    def _fallback_windows(self, video_path):
        """
        如果 TransNet 失败，使用等间隔切片
        """
        print(f"⚠️ [EventGraph] Using fallback windows for {os.path.basename(video_path)}")
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration = count / fps if fps > 0 else 0
        except:
            duration = 0
        
        if duration == 0:
            # 尝试用decord获取时长
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                duration = len(vr) / vr.get_avg_fps()
            except:
                return [(0.0, 1.0)] # 最后的兜底
        
        events = []
        # 每2秒切一段 (比之前的逻辑稍微密集一点，保证覆盖)
        step = 2.0
        for t in np.arange(0, duration, step):
            events.append((t, min(t + step, duration)))
        
        # 如果视频极短
        if not events:
            events = [(0.0, min(1.0, duration))]
            
        return events

    def _build_graph_cot_prompt(self, question, options, segments, adj_matrix, selected_indices):
        event_timeline = [f"Event{i+1}" for i, _, _ in segments]
        
        # 格式化选项字符串
        if isinstance(options, list):
            # 处理可能是字典的情况
            options_clean = []
            for opt in options:
                if isinstance(opt, dict):
                    options_clean.append(str(opt))
                else:
                    options_clean.append(str(opt))
            
            # 如果是A,B,C,D格式
            if len(options_clean) > 0 and (options_clean[0].startswith('A') or len(options_clean) <= 5):
                 options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options_clean)])
            else:
                 options_str = "\n".join(options_clean)
        else:
            options_str = str(options)

        prompt = (
            f"Question: {question}\n"
            f"Options:\n{options_str}\n"
            f"Key Events Timeline: {' -> '.join(event_timeline)}\n"
            f"Based on these visual events, reason step-by-step and choose the best answer."
        )
        return prompt

    def process_and_inference(self, video_path, question, options):
        # 1. 检测事件 (TransNet V2)
        events = self._detect_shot_boundaries(video_path)
        if not events: return "C"
        
        # 2. 提取特征
        global_feats, local_feats, frames = self._extract_event_features(video_path, events)
        if len(frames) == 0: return "C"

        # 3. 建图
        Pi = self._construct_event_graph(global_feats, local_feats, events)
        
        # 4. 选图
        sel_idx = self._select_subgraph(Pi, question, global_feats, events)
        if not sel_idx: sel_idx = [0] # Fallback
        
        # 5. 准备推理数据
        # 确保索引不越界
        valid_sel_idx = [i for i in sel_idx if i < len(frames)]
        if not valid_sel_idx: valid_sel_idx = [0]
        
        selected_frames = [frames[i] for i in sorted(valid_sel_idx)]
        selected_segments = [(events[i][0], events[i][1], i) for i in sorted(valid_sel_idx)]
        
        # 6. 生成 Prompt
        prompt = self._build_graph_cot_prompt(
            question, options, 
            selected_segments, 
            Pi, valid_sel_idx
        )
        
        # 7. 调用 VLM 推理
        return self.model.generate(selected_frames, prompt, options)