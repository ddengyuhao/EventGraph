# # # /root/icml2026/event_graph/code/eventgraph.py
# # import torch
# # import numpy as np
# # import cv2  # <--- 修复报错: 之前缺少这个导入
# # import os
# # from PIL import Image
# # from transformers import CLIPProcessor, CLIPModel
# # from .base_method import BaseMethod
# # from .graph_builder import compute_similarity_matrix, compute_pagerank_matrix
# # from .celf_solver import CELFSelector
# # from .uboco_detector import UbocoDetector

# # try:
# #     from decord import VideoReader, cpu
# # except ImportError:
# #     print("⚠️ Warning: decord not installed")
# #     VideoReader = None

# # class EventGraphLMM(BaseMethod):
# #     def __init__(self, args, model):
# #         super().__init__(args, model)
        
# #         # Params from Paper Section 4.1
# #         self.tau = 30.0  
# #         self.delta = 0.65 
# #         self.alpha = 0.15 
# #         self.lambda_param = 1.0 
# #         self.token_budget = args.token_budget
        
# #         # Detect token density for budget calculation
# #         backbone_name = getattr(args, 'backbone', '')
# #         if '34B' in backbone_name:
# #             self.tokens_per_frame = 576 
# #         elif 'Qwen' in backbone_name:
# #             self.tokens_per_frame = 256 
# #         else:
# #             self.tokens_per_frame = 256 # Video-LLaVA-7B default
            
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# #         # 1. Load CLIP Once
# #         self._load_clip_model()
        
# #         # 2. Initialize Uboco with the SHARED model
# #         self.shot_detector = UbocoDetector(
# #             device=self.device, 
# #             clip_model=self.clip_model, 
# #             clip_processor=self.clip_processor
# #         )

# #     def _load_clip_model(self):
# #         # Uses local path if available to save download time
# #         local_path = "/root/hhq/models/clip-vit-large-patch14"
# #         model_name = local_path if os.path.exists(local_path) else "openai/clip-vit-large-patch14"
        
# #         try:
# #             self.clip_processor = CLIPProcessor.from_pretrained(model_name)
# #             self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
# #             self.clip_model.eval()
# #         except Exception as e:
# #             print(f"Warning: Loading CLIP from {model_name} failed ({e}), trying openai default.")
# #             self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# #             self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
# #             self.clip_model.eval()

# #     def _detect_shot_boundaries(self, video_path):
# #         # Delegate to the shared-model Uboco detector
# #         try:
# #             # Uboco return timestamps (e.g., [2.5, 5.1, ...])
# #             boundaries = self.shot_detector.detect(video_path, sample_rate=2) 
# #             # Convert boundaries to event intervals (start, end)
# #             events = self._boundaries_to_events(boundaries, video_path)
            
# #             # Filter noise < 1s
# #             events = [e for e in events if (e[1] - e[0]) >= 1.0]
            
# #             # If too few events, use fallback
# #             if len(events) < 3: 
# #                 return self._fallback_windows(video_path)
# #             return events
# #         except Exception as e:
# #             print(f"  [EventGraph] Uboco failed ({e}), using fallback.")
# #             return self._fallback_windows(video_path)

# #     def _boundaries_to_events(self, boundaries, video_path):
# #         """
# #         修复报错: 将时间戳列表转换为 (start, end) 元组列表
# #         """
# #         # 获取视频总时长
# #         cap = cv2.VideoCapture(video_path)
# #         fps = cap.get(cv2.CAP_PROP_FPS)
# #         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #         duration = frame_count / fps if fps > 0 else 0
# #         cap.release()

# #         if duration == 0: return []

# #         # 构造区间
# #         sorted_bounds = sorted([0.0] + list(boundaries) + [duration])
# #         # 去重
# #         sorted_bounds = sorted(list(set(sorted_bounds)))
        
# #         events = []
# #         for i in range(len(sorted_bounds) - 1):
# #             start = sorted_bounds[i]
# #             end = sorted_bounds[i+1]
# #             events.append((start, end))
# #         return events

# #     def _extract_event_features(self, video_path, events):
# #         # Batch processing for CLIP
# #         if VideoReader is None: raise ImportError("decord required")
# #         vr = VideoReader(video_path, ctx=cpu(0))
# #         fps = vr.get_avg_fps()
        
# #         representative_frames = []
# #         valid_indices = []
        
# #         for idx, (start_t, end_t) in enumerate(events):
# #             mid_t = (start_t + end_t) / 2.0
# #             # 安全检查防止越界
# #             if mid_t * fps >= len(vr): continue
            
# #             frame_idx = min(len(vr) - 1, int(mid_t * fps))
# #             frame_np = vr[frame_idx].asnumpy()
# #             representative_frames.append(Image.fromarray(frame_np))
# #             valid_indices.append(idx)
        
# #         # Batch Process
# #         batch_size = 32
# #         global_feats_list = []
# #         local_feats_list = []
        
# #         with torch.no_grad():
# #             for i in range(0, len(representative_frames), batch_size):
# #                 batch = representative_frames[i : i+batch_size]
# #                 inputs = self.clip_processor(images=batch, return_tensors="pt", padding=True)
# #                 inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
# #                 # Global (CLS)
# #                 g_feats = self.clip_model.get_image_features(**inputs)
# #                 global_feats_list.append(g_feats)
                
# #                 # Local (Patches)
# #                 outputs = self.clip_model.vision_model(**inputs, output_hidden_states=True)
# #                 l_feats = outputs.last_hidden_state[:, 1:, :] # Remove CLS
# #                 local_feats_list.append(l_feats)
        
# #         if len(global_feats_list) == 0:
# #             return torch.tensor([]), torch.tensor([]), []

# #         global_feats = torch.cat(global_feats_list, dim=0)
# #         local_feats = torch.cat(local_feats_list, dim=0)
        
# #         return global_feats, local_feats, representative_frames

# #     def _construct_event_graph(self, global_feats, local_feats, events):
# #         """
# #         构建语义-时序图
# #         """
# #         N = global_feats.shape[0]
# #         # 1. Compute Semantic Adjacency (Eq. 3, 4)
# #         adj_semantic = compute_similarity_matrix(
# #             global_feats, local_feats, 
# #             tau=self.tau, 
# #             event_times=events, 
# #             threshold=self.delta
# #         )
# #         # 2. Compute Reachability (PageRank, Eq. 6)
# #         Pi = compute_pagerank_matrix(adj_semantic, alpha=self.alpha)
# #         return Pi

# #     def _select_subgraph(self, Pi, question, global_feats, events):
# #         """
# #         CELF 算法选择关键子图
# #         """
# #         # 1. Encode Question
# #         inputs = self.clip_processor(text=[question], return_tensors="pt", padding=True)
# #         inputs = {k: v.to(self.device) for k, v in inputs.items()}
# #         with torch.no_grad():
# #             q_feat = self.clip_model.get_text_features(**inputs)
# #             q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
        
# #         # 2. Calculate Query Relevance (Eq. 5)
# #         g_norm = global_feats / global_feats.norm(dim=-1, keepdim=True)
# #         relevance = torch.mm(g_norm, q_feat.t()).squeeze() # (N,)
# #         relevance = torch.clamp(relevance, min=0.0) # ReLU
        
# #         # 3. Calculate Cost (Token consumption)
# #         # 简单的线性代价: 每个事件消耗 tokens_per_frame
# #         costs = torch.full((len(events),), self.tokens_per_frame, device=self.device)
        
# #         # 4. CELF Selection
# #         selector = CELFSelector(Pi, relevance, costs, lambda_param=self.lambda_param)
# #         selected_indices = selector.select(budget=self.token_budget)
        
# #         return selected_indices

# #     def _fallback_windows(self, video_path):
# #         """
# #         如果 Uboco 失败，使用等间隔切片
# #         """
# #         cap = cv2.VideoCapture(video_path)
# #         fps = cap.get(cv2.CAP_PROP_FPS)
# #         count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #         cap.release()
        
# #         duration = count / fps if fps > 0 else 0
# #         events = []
# #         # 每2秒切一段
# #         step = 2.0
# #         for t in np.arange(0, duration, step):
# #             events.append((t, min(t + step, duration)))
        
# #         # 如果视频极短或者读不到，给一个默认
# #         if not events:
# #             events = [(0.0, 1.0)]
            
# #         return events

# #     def _build_graph_cot_prompt(self, question, options, segments, adj_matrix, selected_indices):
# #         event_timeline = [f"Event{i+1}" for i, _, _ in segments]
        
# #         # 格式化选项字符串
# #         if isinstance(options, list):
# #             options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
# #         else:
# #             options_str = str(options)

# #         prompt = (
# #             f"Question: {question}\n"
# #             f"Options:\n{options_str}\n"
# #             f"Key Events Timeline: {' -> '.join(event_timeline)}\n"
# #             f"Based on these visual events, reason step-by-step and choose the best answer."
# #         )
# #         return prompt

# #     def process_and_inference(self, video_path, question, options):
# #         # 1. 检测事件
# #         events = self._detect_shot_boundaries(video_path)
# #         if not events: return "C"
        
# #         # 2. 提取特征
# #         global_feats, local_feats, frames = self._extract_event_features(video_path, events)
# #         if len(frames) == 0: return "C"

# #         # 3. 建图
# #         Pi = self._construct_event_graph(global_feats, local_feats, events)
        
# #         # 4. 选图
# #         sel_idx = self._select_subgraph(Pi, question, global_feats, events)
# #         if not sel_idx: sel_idx = [0] # Fallback
        
# #         # 5. 准备推理数据
# #         selected_frames = [frames[i] for i in sorted(sel_idx)]
# #         selected_segments = [(events[i][0], events[i][1], i) for i in sorted(sel_idx)]
        
# #         # 6. 生成 Prompt
# #         prompt = self._build_graph_cot_prompt(
# #             question, options, 
# #             selected_segments, 
# #             Pi, sel_idx
# #         )
        
# #         # 7. 调用 VLM 推理
# #         return self.model.generate(selected_frames, prompt, options)


# # /root/icml2026/event_graph/code/eventgraph.py
# import torch
# import numpy as np
# import cv2
# import os
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# from .base_method import BaseMethod
# from .graph_builder import compute_similarity_matrix, compute_pagerank_matrix
# from .celf_solver import CELFSelector

# # === 修改 1: 导入 TransNetV2Detector ===
# # 假设你把代码保存为了 transnet_detector.py
# try:
#     from .transnet_detector import TransNetV2Detector
# except ImportError:
#     print("⚠️ Warning: Could not import TransNetV2Detector. Make sure transnet_detector.py exists.")
#     TransNetV2Detector = None

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
#         if 'Qwen' in backbone_name:
#             # 策略 A: 强制 Resize 到 336x336 (推荐) -> Token 消耗稳定 ~256
#             self.tokens_per_frame = 256 
#             self.target_size = (336, 336) 
#         elif '34B' in backbone_name:
#             self.tokens_per_frame = 576
#             self.target_size = None # LLaVA-Next 内部处理
#         else:
#             self.tokens_per_frame = 256
#             self.target_size = None
            
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # 1. Load CLIP Once (用于后续特征提取和建图，不用于检测了)
#         self._load_clip_model()
        
#         # === 修改 2: 初始化 TransNet V2 ===
#         if TransNetV2Detector is not None:
#             print("🚀 [EventGraph] Initializing TransNet V2 Detector...")
#             self.shot_detector = TransNetV2Detector(device='cuda') # 强制使用cuda如果可用
#         else:
#             self.shot_detector = None
#             print("❌ [EventGraph] TransNet V2 Detector not available. Will use fallback.")

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
#         """
#         使用 TransNet V2 进行镜头分割
#         """
#         # 如果检测器没初始化，直接回退
#         if self.shot_detector is None:
#             return self._fallback_windows(video_path)

#         try:
#             # === 修改 3: 直接调用 detect_shots ===
#             # TransNetV2Detector.detect_shots 已经返回了 [(start, end), ...] 格式
#             # 不需要再手动转换 boundaries_to_events
#             events = self.shot_detector.detect_shots(video_path, threshold=0.5)
            
#             # 过滤极短的噪声片段 (< 0.5s)
#             events = [e for e in events if (e[1] - e[0]) >= 0.5]
            
#             # 如果检测到的事件太少，说明可能是长镜头或者检测失败，使用回退策略
#             if len(events) < 1: 
#                 print(f"  [EventGraph] Too few shots detected ({len(events)}), using fallback.")
#                 return self._fallback_windows(video_path)
            
#             return events

#         except Exception as e:
#             print(f"  ❌ [EventGraph] TransNet V2 failed ({e}), using fallback.")
#             return self._fallback_windows(video_path)

#     # _boundaries_to_events 函数现在可以删除了，因为 TransNet 类内部处理了
#     # 但为了防止某些子类继承调用，你可以保留它，或者直接删除以保持代码整洁。
#     # 这里我把它移除了。

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
#             if mid_t * fps >= len(vr): 
#                 # 尝试取最后一张
#                 frame_idx = len(vr) - 1
#             else:
#                 frame_idx = min(len(vr) - 1, int(mid_t * fps))
            
#             try:
#                 frame_np = vr[frame_idx].asnumpy()
#                 representative_frames.append(Image.fromarray(frame_np))
#                 valid_indices.append(idx)
#             except Exception as e:
#                 print(f"Error extracting frame at {mid_t}s: {e}")
#                 continue
        
#         if not representative_frames:
#             return torch.tensor([]), torch.tensor([]), []

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
#         # 确保数据在同一设备
#         global_feats = global_feats.to(self.device)
#         local_feats = local_feats.to(self.device)
        
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
#         # Normalize global feats
#         g_norm = global_feats / global_feats.norm(dim=-1, keepdim=True)
#         relevance = torch.mm(g_norm, q_feat.t()).squeeze() # (N,)
        
#         # Handle shape mismatch if only 1 event
#         if relevance.dim() == 0:
#             relevance = relevance.unsqueeze(0)
            
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
#         如果 TransNet 失败，使用等间隔切片
#         """
#         print(f"⚠️ [EventGraph] Using fallback windows for {os.path.basename(video_path)}")
#         try:
#             cap = cv2.VideoCapture(video_path)
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             cap.release()
#             duration = count / fps if fps > 0 else 0
#         except:
#             duration = 0
        
#         if duration == 0:
#             # 尝试用decord获取时长
#             try:
#                 vr = VideoReader(video_path, ctx=cpu(0))
#                 duration = len(vr) / vr.get_avg_fps()
#             except:
#                 return [(0.0, 1.0)] # 最后的兜底
        
#         events = []
#         # 每2秒切一段 (比之前的逻辑稍微密集一点，保证覆盖)
#         step = 2.0
#         for t in np.arange(0, duration, step):
#             events.append((t, min(t + step, duration)))
        
#         # 如果视频极短
#         if not events:
#             events = [(0.0, min(1.0, duration))]
            
#         return events

#     # def _build_graph_cot_prompt(self, question, options, segments, adj_matrix, selected_indices):
#     #     event_timeline = [f"Event{i+1}" for i, _, _ in segments]
        
#     #     # 格式化选项字符串
#     #     if isinstance(options, list):
#     #         # 处理可能是字典的情况
#     #         options_clean = []
#     #         for opt in options:
#     #             if isinstance(opt, dict):
#     #                 options_clean.append(str(opt))
#     #             else:
#     #                 options_clean.append(str(opt))
            
#     #         # 如果是A,B,C,D格式
#     #         if len(options_clean) > 0 and (options_clean[0].startswith('A') or len(options_clean) <= 5):
#     #              options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options_clean)])
#     #         else:
#     #              options_str = "\n".join(options_clean)
#     #     else:
#     #         options_str = str(options)

#     #     prompt = (
#     #         f"Question: {question}\n"
#     #         f"Options:\n{options_str}\n"
#     #         f"Key Events Timeline: {' -> '.join(event_timeline)}\n"
#     #         f"Based on these visual events, reason step-by-step and choose the best answer."
#     #     )
#     #     return prompt
#     def _build_graph_cot_prompt(self, question, options, segments, adj_matrix, selected_indices):
#         # 1. Build a structured timeline with timestamps
#         timeline_str = ""
#         for idx, (start, end, original_idx) in enumerate(segments):
#             # Add explicit "Node" markers
#             timeline_str += f"- Node {idx+1} (Time: {start:.1f}s - {end:.1f}s): [Visual Content]\n"

#         # 2. Add "Graph Hints" (Optional: Tell the LLM which nodes are semantically related)
#         # We look at the adjacency matrix for selected nodes to find strong non-temporal links
#         graph_hints = []
#         for i in range(len(selected_indices)):
#             for j in range(len(selected_indices)):
#                 if i == j: continue
#                 # original graph indices
#                 u, v = selected_indices[i], selected_indices[j]
#                 # If there was a strong semantic edge in the original graph
#                 if adj_matrix[u, v] > 0.05: # Threshold for hint
#                     graph_hints.append(f"Node {i+1} is semantically related to Node {j+1}.")
        
#         hints_str = "\n".join(graph_hints[:5]) # Limit hints to avoid noise

#         # 3. Format Options
#         if isinstance(options, list):
#             options_clean = [str(o) for o in options]
#             options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options_clean)])
#         else:
#             options_str = str(options)

#         # 4. Structured CoT Prompt
#         prompt = (
#             f"You are analyzing a long video. I have selected key events for you based on a semantic graph.\n\n"
#             f"User Query: {question}\n\n"
#             f"Selected Key Events Timeline:\n{timeline_str}\n"
#             f"Key Semantic Connections identified by the graph:\n{hints_str}\n\n"
#             f"Options:\n{options_str}\n\n"
#             f"Instructions:\n"
#             f"1. Analyze the visual content of each Node relevant to the query.\n"
#             f"2. Connect the clues: If Node X and Node Y are related, combine their information.\n"
#             f"3. Reason step-by-step to answer the query.\n"
#             f"Answer:"
#         )
#         return prompt

#     def _build_simple_prompt(self, question, options):
#         """构建简单的 QA Prompt，不需要 Event Timeline"""
#         # 格式化选项
#         if isinstance(options, list) and options:
#             # 清洗选项，确保都是字符串
#             clean_opts = []
#             for opt in options:
#                 clean_opts.append(str(opt))
                
#             options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(clean_opts)])
#             prompt = f"{question}\nOptions:\n{options_str}\nAnswer with the option letter directly."
#         else:
#             # 开放式问题
#             prompt = f"{question}\nAnswer the question in detail."
            
#         return prompt

#     def process_and_inference(self, video_path, question, options):
#         # 1. 检测事件 (TransNet V2)
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
#         valid_sel_idx = [i for i in sel_idx if i < len(frames)]
#         if not valid_sel_idx: valid_sel_idx = [0]
        
#         # --- IMPROVEMENT: Sort indices strictly by time ---
#         valid_sel_idx = sorted(valid_sel_idx)

#         # --- IMPROVEMENT: Force Resize to ensure Token Budget fits more frames ---
#         # For Qwen/LLaVA, 336x336 usually takes ~256 tokens. 
#         # This allows you to fit ~16 frames in a 4k budget, covering more timeline.
#         target_resolution = (336, 336) 
        
#         selected_frames = []
#         for i in valid_sel_idx:
#             img = frames[i]
#             # Resize guarantees token count matches your self.tokens_per_frame estimation
#             img_resized = img.resize(target_resolution, Image.BICUBIC)
#             selected_frames.append(img_resized)

#         selected_segments = [(events[i][0], events[i][1], i) for i in valid_sel_idx]
        
#         # 6. 生成 Prompt (稍微加强一下 Prompt，让它明确输出)
#         # 建议在 prompt 最后加一句明确的指令
#         prompt = self._build_graph_cot_prompt(
#             question, options, 
#             selected_segments, 
#             Pi, valid_sel_idx
#         )
#         prompt += "\nImportant: End your response with 'The answer is X.'"

#         # =======
#         # prompt = self._build_simple_prompt(question, options)

#         # 7. 调用 VLM 推理
#         # 🔥 修改这里：显式传入 max_new_tokens
#         # Video-MME 的推理通常需要较长篇幅，建议设为 1024 或 2048
#         return self.model.generate(
#             selected_frames, 
#             prompt, 
#             options, 
#             max_new_tokens=40960  # <--- 增加这个参数
#         )



# # # import torch
# # # import numpy as np
# # # import cv2
# # # import os
# # # from PIL import Image
# # # from transformers import CLIPProcessor, CLIPModel
# # # from .base_method import BaseMethod
# # # from .graph_builder import compute_similarity_matrix, compute_pagerank_matrix
# # # from .celf_solver import CELFSelector

# # # try:
# # #     from .transnet_detector import TransNetV2Detector
# # # except ImportError:
# # #     # print("⚠️ Warning: Could not import TransNetV2Detector.")
# # #     TransNetV2Detector = None

# # # try:
# # #     from decord import VideoReader, cpu
# # # except ImportError:
# # #     print("⚠️ Warning: decord not installed")
# # #     VideoReader = None

# # # class EventGraphLMM(BaseMethod):
# # #     def __init__(self, args, model):
# # #         super().__init__(args, model)
        
# # #         # Params
# # #         self.tau = 30.0  
# # #         self.delta = 0.65 
# # #         self.alpha = 0.15 
# # #         self.lambda_param = 1.0 
# # #         self.token_budget = args.token_budget
        
# # #         # Detect token density
# # #         backbone_name = getattr(args, 'backbone', '')
# # #         if 'Qwen' in backbone_name:
# # #             self.tokens_per_frame = 256 
# # #             self.target_size = (336, 336) 
# # #         elif '34B' in backbone_name:
# # #             self.tokens_per_frame = 576
# # #             self.target_size = None 
# # #         else:
# # #             self.tokens_per_frame = 256
# # #             self.target_size = None
            
# # #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# # #         # Load CLIP
# # #         self._load_clip_model()
        
# # #         # Init TransNet
# # #         if TransNetV2Detector is not None:
# # #             # print("🚀 [EventGraph] Initializing TransNet V2...")
# # #             self.shot_detector = TransNetV2Detector(device='cuda')
# # #         else:
# # #             self.shot_detector = None

# # #     def _load_clip_model(self):
# # #         local_path = "/root/hhq/models/clip-vit-large-patch14"
# # #         model_name = local_path if os.path.exists(local_path) else "openai/clip-vit-large-patch14"
# # #         try:
# # #             self.clip_processor = CLIPProcessor.from_pretrained(model_name)
# # #             self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
# # #             self.clip_model.eval()
# # #         except Exception as e:
# # #             print(f"Warning: Loading CLIP failed, using default. {e}")
# # #             self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# # #             self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
# # #             self.clip_model.eval()

# # #     def _detect_shot_boundaries(self, video_path):
# # #         if self.shot_detector is None:
# # #             return self._fallback_windows(video_path)

# # #         try:
# # #             events = self.shot_detector.detect_shots(video_path, threshold=0.5)
# # #             # Filter noise < 0.5s
# # #             events = [e for e in events if (e[1] - e[0]) >= 0.5]
            
# # #             # LVBench 优化: 如果检测出的镜头过多 (>600)，进行合并或降采样
# # #             # 防止 Graph 构建过慢
# # #             if len(events) > 600:
# # #                 # 简单策略：每隔一个取一个，或者合并相邻
# # #                 events = events[::2] 

# # #             if len(events) < 1: 
# # #                 return self._fallback_windows(video_path)
# # #             return events

# # #         except Exception as e:
# # #             print(f"  ❌ TransNet error: {e}")
# # #             return self._fallback_windows(video_path)

# # #     def _extract_event_features(self, video_path, events):
# #         # if VideoReader is None: raise ImportError("decord required")
        
# #         # # 显存优化: 强制使用 CPU 读取视频，防止 CUDA 初始化冲突
# #         # try:
# #         #     vr = VideoReader(video_path, ctx=cpu(0))
# #         # except Exception as e:
# #         #     print(f"❌ Decord Init Failed: {e}")
# #         #     return torch.tensor([]), torch.tensor([]), []
            
# #         # fps = vr.get_avg_fps()
# #         # total_frames = len(vr)
        
# #         # representative_frames = []
# #         # valid_indices = []
        
# #         # # 1. 读取帧
# #         # for idx, (start_t, end_t) in enumerate(events):
# #         #     mid_t = (start_t + end_t) / 2.0
# #         #     frame_idx = int(mid_t * fps)
# #         #     if frame_idx >= total_frames: frame_idx = total_frames - 1
            
# #         #     try:
# #         #         frame_np = vr[frame_idx].asnumpy()
# #         #         representative_frames.append(Image.fromarray(frame_np))
# #         #         valid_indices.append(idx)
# #         #     except:
# #         #         continue
        
# #         # # 显式释放 Decord 资源
# #         # del vr
        
# #         # if not representative_frames:
# #         #     return torch.tensor([]), torch.tensor([]), []

# #         # # 2. 批量提取特征
# #         # batch_size = 16 
# #         # global_feats_list = []
# #         # local_feats_list = []
        
# #         # with torch.no_grad():
# #         #     for i in range(0, len(representative_frames), batch_size):
# #         #         batch = representative_frames[i : i+batch_size]
# #         #         inputs = self.clip_processor(images=batch, return_tensors="pt", padding=True)
# #         #         inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
# #         #         # --- Global Features ---
# #         #         # get_image_features 通常返回 Tensor，但防御性编程防止它返回 ModelOutput
# #         #         g_feats = self.clip_model.get_image_features(**inputs)
                
# #         #         # 🔥 修复：如果返回的是对象而不是 Tensor，提取其中的 Tensor
# #         #         if not isinstance(g_feats, torch.Tensor):
# #         #             if hasattr(g_feats, 'image_embeds'): # 新版 Transformers 可能的字段
# #         #                 g_feats = g_feats.image_embeds
# #         #             elif hasattr(g_feats, 'pooler_output'):
# #         #                 g_feats = g_feats.pooler_output
                
# #         #         # --- Local Features ---
# #         #         # vision_model 返回的是 BaseModelOutputWithPooling
# #         #         outputs = self.clip_model.vision_model(**inputs, output_hidden_states=True)
                
# #         #         if hasattr(outputs, 'last_hidden_state'):
# #         #             l_feats = outputs.last_hidden_state[:, 1:, :] # 去掉 CLS token
# #         #         elif isinstance(outputs, tuple):
# #         #             l_feats = outputs[0][:, 1:, :]
# #         #         else:
# #         #             raise ValueError(f"Unknown output type from vision_model: {type(outputs)}")
                
# #         #         # 转 CPU 释放显存
# #         #         global_feats_list.append(g_feats.cpu())
# #         #         local_feats_list.append(l_feats.cpu())
        
# #         # if len(global_feats_list) == 0:
# #         #     return torch.tensor([]), torch.tensor([]), []

# #         # global_feats = torch.cat(global_feats_list, dim=0)
# #         # local_feats = torch.cat(local_feats_list, dim=0)
        
# #         # return global_feats, local_feats, representative_frames

# # #     def _construct_event_graph(self, global_feats, local_feats, events):
# # #         """
# # #         构建图，支持 CPU Offload
# # #         """
# # #         N = global_feats.shape[0]
        
# # #         # 如果节点太多，强制使用 CPU 计算矩阵，避免 OOM
# # #         # 对于 LVBench，N 可能达到 500+，N^2 矩阵还行，但中间变量大
# # #         compute_device = self.device
# # #         if N > 300: 
# # #             compute_device = torch.device('cpu')

# # #         global_feats = global_feats.to(compute_device)
# # #         local_feats = local_feats.to(compute_device)

# # #         # 1. Compute Semantic Adjacency
# # #         # 修改 graph_builder 里的函数让它接受 device 参数 (如果支持)
# # #         # 或者确保它是纯 PyTorch 操作，会自动跟随 tensor 的 device
# # #         try:
# # #             adj_semantic = compute_similarity_matrix(
# # #                 global_feats, local_feats, 
# # #                 tau=self.tau, 
# # #                 event_times=events, 
# # #                 threshold=self.delta
# # #             )
# # #         except RuntimeError:
# # #             # 如果 GPU 爆了，回退到 CPU
# # #             print("⚠️ Graph construction OOM, switching to CPU.")
# # #             global_feats = global_feats.cpu()
# # #             local_feats = local_feats.cpu()
# # #             adj_semantic = compute_similarity_matrix(
# # #                 global_feats, local_feats, 
# # #                 tau=self.tau, 
# # #                 event_times=events, 
# # #                 threshold=self.delta
# # #             )
            
# # #         # 2. PageRank
# # #         Pi = compute_pagerank_matrix(adj_semantic, alpha=self.alpha)
        
# # #         # 结果转回 GPU (如果 CELF 需要 GPU) 或保持 CPU
# # #         return Pi.to(self.device)

# # #     def _select_subgraph(self, Pi, question, global_feats, events):
# # #         # Encode Question
# # #         inputs = self.clip_processor(text=[question], return_tensors="pt", padding=True)
# # #         inputs = {k: v.to(self.device) for k, v in inputs.items()}
# # #         with torch.no_grad():
# # #             q_feat = self.clip_model.get_text_features(**inputs)
# # #             q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
        
# # #         # Relevance calculation
# # #         # 确保 global_feats 在 GPU 上进行矩阵乘法 (比较快)
# # #         # 如果显存极度紧张，可以把 q_feat 转 CPU
# # #         g_feat_dev = global_feats.to(self.device)
# # #         g_norm = g_feat_dev / g_feat_dev.norm(dim=-1, keepdim=True)
        
# # #         relevance = torch.mm(g_norm, q_feat.t()).squeeze()
# # #         if relevance.dim() == 0: relevance = relevance.unsqueeze(0)
# # #         relevance = torch.clamp(relevance, min=0.0)
        
# # #         # Costs
# # #         costs = torch.full((len(events),), self.tokens_per_frame, device=self.device)
        
# # #         # CELF
# # #         # 确保 Pi 也在 device
# # #         Pi = Pi.to(self.device)
# # #         selector = CELFSelector(Pi, relevance, costs, lambda_param=self.lambda_param)
# # #         selected_indices = selector.select(budget=self.token_budget)
        
# # #         return selected_indices

# # #     def _fallback_windows(self, video_path):
# # #         try:
# # #             cap = cv2.VideoCapture(video_path)
# # #             fps = cap.get(cv2.CAP_PROP_FPS)
# # #             count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# # #             cap.release()
# # #             duration = count / fps if fps > 0 else 0
# # #         except:
# # #             duration = 0
            
# # #         if duration == 0: return [(0.0, 1.0)]
        
# # #         events = []
# # #         # LVBench 优化: 动态步长
# # #         # 视频越长，切片越稀疏，防止生成几千个片段
# # #         if duration < 300: step = 2.0         # 短视频: 2s
# # #         elif duration < 1800: step = 5.0      # 30分钟内: 5s
# # #         else: step = 10.0                     # 长视频: 10s
        
# # #         for t in np.arange(0, duration, step):
# # #             events.append((t, min(t + step, duration)))
            
# # #         if not events: events = [(0.0, min(1.0, duration))]
# # #         return events

# # #     def _build_simple_prompt(self, question, options):
# # #         if isinstance(options, list) and options:
# # #             clean_opts = [str(opt) for opt in options]
# # #             options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(clean_opts)])
# # #             prompt = f"{question}\nOptions:\n{options_str}\nAnswer with the option letter directly."
# # #         else:
# # #             prompt = f"{question}\nAnswer the question in detail."
# # #         return prompt

# # #     def process_and_inference(self, video_path, question, options):
# # #         # 1. 检测
# # #         events = self._detect_shot_boundaries(video_path)
# # #         if not events: return "C"
        
# # #         # 2. 提取 (已做 CPU Offload)
# # #         global_feats, local_feats, frames = self._extract_event_features(video_path, events)
# # #         if len(frames) == 0: return "C"

# # #         # 3. 建图
# # #         Pi = self._construct_event_graph(global_feats, local_feats, events)
        
# # #         # 4. 选图
# # #         sel_idx = self._select_subgraph(Pi, question, global_feats, events)
# # #         if not sel_idx: sel_idx = [0]
        
# # #         # 5. 准备推理数据
# # #         valid_sel_idx = [i for i in sel_idx if i < len(frames)]
# # #         if not valid_sel_idx: valid_sel_idx = [0]
        
# # #         # 排序
# # #         valid_sel_idx = sorted(valid_sel_idx)

# # #         # 🔥 LVBench 关键优化: 推理帧数硬截断 (Hard Cap)
# # #         # 即使 CELF 选了 100 张，我们也只取 Top-K 或者均匀采样到 K
# # #         # 防止 Qwen 爆显存
# # #         MAX_INFERENCE_FRAMES = 64
# # #         if len(valid_sel_idx) > MAX_INFERENCE_FRAMES:
# # #             # 简单的均匀降采样
# # #             indices = np.linspace(0, len(valid_sel_idx) - 1, MAX_INFERENCE_FRAMES).astype(int)
# # #             valid_sel_idx = [valid_sel_idx[i] for i in indices]

# # #         selected_frames = []
# # #         for i in valid_sel_idx:
# # #             img = frames[i]
# # #             # Resize
# # #             if self.target_size:
# # #                 img_resized = img.resize(self.target_size, Image.BICUBIC)
# # #                 selected_frames.append(img_resized)
# # #             else:
# # #                 selected_frames.append(img)

# # #         # 6. Prompt
# # #         prompt = self._build_simple_prompt(question, options)

# # #         # 7. 推理
# # #         # max_new_tokens 设为 1024 (足够回答 A/B/C/D 或简短 CoT)
# # #         # 之前 40960 会直接爆显存
# # #         return self.model.generate(
# # #             selected_frames, 
# # #             prompt, 
# # #             options, 
# # #             max_new_tokens=1024  
# # #         )

# # import torch
# # import numpy as np
# # import cv2
# # import os
# # from PIL import Image
# # from transformers import CLIPProcessor, CLIPModel
# # from .base_method import BaseMethod
# # from .graph_builder import compute_similarity_matrix, compute_pagerank_matrix
# # from .celf_solver import CELFSelector

# # try:
# #     from .transnet_detector import TransNetV2Detector
# # except ImportError:
# #     TransNetV2Detector = None

# # try:
# #     from decord import VideoReader, cpu, gpu
# # except ImportError:
# #     print("⚠️ Warning: decord not installed")
# #     VideoReader = None

# # class EventGraphLMM(BaseMethod):
# #     def __init__(self, args, model):
# #         super().__init__(args, model)
        
# #         # 参数
# #         self.tau = 30.0  
# #         self.delta = 0.65 
# #         self.alpha = 0.15 
# #         self.lambda_param = 1.0 
# #         self.token_budget = args.token_budget
        
# #         # Token 估算
# #         backbone_name = getattr(args, 'backbone', '')
# #         if 'Qwen' in backbone_name:
# #             self.tokens_per_frame = 256 
# #             self.target_size = (336, 336) 
# #         elif '34B' in backbone_name:
# #             self.tokens_per_frame = 576
# #             self.target_size = None 
# #         else:
# #             self.tokens_per_frame = 256
# #             self.target_size = None
            
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# #         # 加载 CLIP
# #         self._load_clip_model()
        
# #         # 初始化 TransNet
# #         if TransNetV2Detector is not None:
# #             # 🔥 强制 CPU 模式！
# #             # 1. 彻底避开 cuDNN 版本冲突（TensorFlow CPU 版不需要 cuDNN）
# #             # 2. 72B 模型需要所有显存，镜头检测这种小任务交给 CPU 绰绰有余
# #             print("🚀 [EventGraph] Initializing TransNet V2 on CPU (Safe Mode)...")
# #             try:
# #                 self.shot_detector = TransNetV2Detector(device='cuda')
# #             except Exception as e:
# #                 print(f"⚠️ TransNet Init Failed: {e}. Will use fallback windows.")
# #                 self.shot_detector = None
# #         else:
# #             self.shot_detector = None

# #     def _load_clip_model(self):
# #         local_path = "/root/hhq/models/clip-vit-large-patch14"
# #         model_name = local_path if os.path.exists(local_path) else "openai/clip-vit-large-patch14"
# #         try:
# #             self.clip_processor = CLIPProcessor.from_pretrained(model_name)
# #             self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
# #             self.clip_model.eval()
# #         except Exception as e:
# #             print(f"Warning: Loading CLIP failed ({e}), using default.")
# #             self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# #             self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
# #             self.clip_model.eval()

# #     def _detect_shot_boundaries(self, video_path):
# #         if self.shot_detector is None:
# #             return self._fallback_windows(video_path)
# #         try:
# #             # 这里的 threshold 可以稍微调低一点以获得更多细粒度事件
# #             events = self.shot_detector.detect_shots(video_path, threshold=0.3)
# #             events = [e for e in events if (e[1] - e[0]) >= 0.5]
            
# #             # LVBench 长视频优化：限制最大事件数
# #             if len(events) > 1000:
# #                 events = events[::2] 

# #             if len(events) < 1: 
# #                 return self._fallback_windows(video_path)
# #             return events
# #         except Exception as e:
# #             print(f"  ❌ TransNet error: {e}")
# #             return self._fallback_windows(video_path)

# #     def _extract_event_features(self, video_path, events):
# #         if VideoReader is None: raise ImportError("decord required")
        
# #         try:
# #             vr = VideoReader(video_path, ctx=cpu(0))
# #         except:
# #             return torch.tensor([]), torch.tensor([]), []
            
# #         fps = vr.get_avg_fps()
# #         total_frames = len(vr)
        
# #         representative_frames = []
        
# #         # 1. 读取帧 (CPU -> RAM)
# #         for idx, (start_t, end_t) in enumerate(events):
# #             mid_t = (start_t + end_t) / 2.0
# #             frame_idx = int(mid_t * fps)
# #             if frame_idx >= total_frames: frame_idx = total_frames - 1
            
# #             try:
# #                 frame_np = vr[frame_idx].asnumpy()
# #                 representative_frames.append(Image.fromarray(frame_np))
# #             except:
# #                 continue
        
# #         del vr 
# #         if not representative_frames:
# #             return torch.tensor([]), torch.tensor([]), []

# #         # 2. 批量提取特征
# #         batch_size = 64
# #         global_feats_list = []
# #         local_feats_list = []
        
# #         with torch.no_grad():
# #             for i in range(0, len(representative_frames), batch_size):
# #                 batch = representative_frames[i : i+batch_size]
# #                 inputs = self.clip_processor(images=batch, return_tensors="pt", padding=True)
# #                 inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
# #                 # Global
# #                 g_out = self.clip_model.get_image_features(**inputs)
# #                 # 类型检查
# #                 if isinstance(g_out, torch.Tensor):
# #                     g_feats = g_out
# #                 elif hasattr(g_out, 'image_embeds'):
# #                     g_feats = g_out.image_embeds
# #                 elif hasattr(g_out, 'pooler_output'):
# #                     g_feats = g_out.pooler_output
# #                 else:
# #                     g_feats = g_out[0]
                
# #                 # Local
# #                 l_out = self.clip_model.vision_model(**inputs, output_hidden_states=True)
# #                 # 类型检查
# #                 if hasattr(l_out, 'last_hidden_state'):
# #                     l_feats = l_out.last_hidden_state[:, 1:, :] 
# #                 elif isinstance(l_out, tuple):
# #                     l_feats = l_out[0][:, 1:, :]
# #                 else:
# #                     l_feats = g_feats.unsqueeze(1)

# #                 global_feats_list.append(g_feats) 
# #                 local_feats_list.append(l_feats)
        
# #         if len(global_feats_list) == 0:
# #             return torch.tensor([]), torch.tensor([]), []

# #         global_feats = torch.cat(global_feats_list, dim=0)
# #         local_feats = torch.cat(local_feats_list, dim=0)
        
# #         return global_feats, local_feats, representative_frames

# #     def _construct_event_graph(self, global_feats, local_feats, events):
# #         # 此时数据已经在 GPU 上了，直接算
# #         adj_semantic = compute_similarity_matrix(
# #             global_feats, local_feats, 
# #             tau=self.tau, 
# #             event_times=events, 
# #             threshold=self.delta
# #         )
# #         Pi = compute_pagerank_matrix(adj_semantic, alpha=self.alpha)
# #         return Pi

# #     def _select_subgraph(self, Pi, question, global_feats, events):
# #         # 1. Encode Question
# #         inputs = self.clip_processor(text=[question], return_tensors="pt", padding=True)
# #         inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
# #         with torch.no_grad():
# #             q_out = self.clip_model.get_text_features(**inputs)
            
# #             # 🔥 [关键修复]：针对 Text Features 的类型检查
# #             if isinstance(q_out, torch.Tensor):
# #                 q_feat = q_out
# #             elif hasattr(q_out, 'text_embeds'):
# #                 q_feat = q_out.text_embeds
# #             elif hasattr(q_out, 'pooler_output'):
# #                 q_feat = q_out.pooler_output
# #             else:
# #                 q_feat = q_out[0] # Tuple fallback
            
# #             # 现在 q_feat 肯定是 Tensor，可以安全调用 norm
# #             q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
        
# #         # 2. Relevance calculation (GPU)
# #         g_norm = global_feats / global_feats.norm(dim=-1, keepdim=True)
# #         relevance = torch.mm(g_norm, q_feat.t()).squeeze()
# #         if relevance.dim() == 0: relevance = relevance.unsqueeze(0)
# #         relevance = torch.clamp(relevance, min=0.0)
        
# #         costs = torch.full((len(events),), self.tokens_per_frame, device=self.device)
        
# #         # 3. CELF Selection
# #         selector = CELFSelector(Pi, relevance, costs, lambda_param=self.lambda_param)
# #         selected_indices = selector.select(budget=self.token_budget)
        
# #         return selected_indices

# #     def _fallback_windows(self, video_path):
# #         try:
# #             cap = cv2.VideoCapture(video_path)
# #             fps = cap.get(cv2.CAP_PROP_FPS)
# #             count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #             cap.release()
# #             duration = count / fps if fps > 0 else 0
# #         except:
# #             duration = 0
            
# #         if duration == 0: return [(0.0, 1.0)]
        
# #         events = []
# #         if duration < 300: step = 2.0
# #         elif duration < 1800: step = 5.0
# #         else: step = 10.0
        
# #         for t in np.arange(0, duration, step):
# #             events.append((t, min(t + step, duration)))
            
# #         if not events: events = [(0.0, min(1.0, duration))]
# #         return events

# #     def _build_simple_prompt(self, question, options):
# #         if isinstance(options, list) and options:
# #             clean_opts = [str(opt) for opt in options]
# #             options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(clean_opts)])
# #             prompt = f"{question}\nOptions:\n{options_str}\nAnswer with the option letter directly."
# #         else:
# #             prompt = f"{question}\nAnswer the question in detail."
# #         return prompt

# #     def process_and_inference(self, video_path, question, options):
# #         # 1. 检测
# #         events = self._detect_shot_boundaries(video_path)
# #         if not events: return "C"
        
# #         # 2. 提取
# #         global_feats, local_feats, frames = self._extract_event_features(video_path, events)
# #         if len(frames) == 0: return "C"

# #         # 3. 建图
# #         Pi = self._construct_event_graph(global_feats, local_feats, events)
        
# #         # 4. 选图
# #         sel_idx = self._select_subgraph(Pi, question, global_feats, events)
# #         if not sel_idx: sel_idx = [0]
        
# #         valid_sel_idx = [i for i in sel_idx if i < len(frames)]
# #         if not valid_sel_idx: valid_sel_idx = [0]
# #         valid_sel_idx = sorted(valid_sel_idx)

# #         # 推理帧数优化 (128帧)
# #         MAX_INFERENCE_FRAMES = 128
# #         if len(valid_sel_idx) > MAX_INFERENCE_FRAMES:
# #             indices = np.linspace(0, len(valid_sel_idx) - 1, MAX_INFERENCE_FRAMES).astype(int)
# #             valid_sel_idx = [valid_sel_idx[i] for i in indices]

# #         selected_frames = []
# #         for i in valid_sel_idx:
# #             img = frames[i]
# #             if self.target_size:
# #                 img_resized = img.resize(self.target_size, Image.BICUBIC)
# #                 selected_frames.append(img_resized)
# #             else:
# #                 selected_frames.append(img)

# #         # 6. Prompt
# #         prompt = self._build_simple_prompt(question, options)

# #         # 7. 推理 (Token 2048)
# #         return self.model.generate(
# #             selected_frames, 
# #             prompt, 
# #             options, 
# #             max_new_tokens=10240
# #         )

import torch
import numpy as np
import cv2
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from .base_method import BaseMethod
from .graph_builder import compute_similarity_matrix, compute_pagerank_matrix
from .celf_solver import CELFSelector

# ✅ 恢复 TransNet 导入
try:
    from .transnet_detector import TransNetV2Detector
except ImportError:
    print("⚠️ Warning: Could not import TransNetV2Detector.")
    TransNetV2Detector = None

try:
    from decord import VideoReader, cpu
except ImportError:
    print("⚠️ Warning: decord not installed")
    VideoReader = None

class EventGraphLMM(BaseMethod):
    def __init__(self, args, model):
        super().__init__(args, model)
        
        # Params
        self.tau = 30.0  
        self.delta = 0.65 
        self.alpha = 0.15 
        self.lambda_param = 1.0 
        self.token_budget = args.token_budget
        
        # Token 估算
        backbone_name = getattr(args, 'backbone', '')
        if 'Qwen' in backbone_name:
            self.tokens_per_frame = 256 
            self.target_size = (336, 336) 
        elif '34B' in backbone_name:
            self.tokens_per_frame = 576
            self.target_size = None 
        else:
            self.tokens_per_frame = 256
            self.target_size = None
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load CLIP
        self._load_clip_model()
        
        # ✅ 2. 初始化 TransNet (你已解决 CUDA 冲突，这里恢复使用 GPU)
        if TransNetV2Detector is not None:
            print("🚀 [EventGraph] Initializing TransNet V2 Detector on GPU...")
            try:
                self.shot_detector = TransNetV2Detector(device='cuda') 
            except Exception as e:
                print(f"❌ TransNet Init Failed: {e}. Using fallback.")
                self.shot_detector = None
        else:
            self.shot_detector = None

    def _load_clip_model(self):
        local_path = "/root/hhq/models/clip-vit-large-patch14"
        model_name = local_path if os.path.exists(local_path) else "openai/clip-vit-large-patch14"
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.clip_model.eval()
        except Exception as e:
            print(f"Warning: Loading CLIP failed ({e}), using default.")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.clip_model.eval()

    def _detect_shot_boundaries(self, video_path):
        """
        使用 TransNet V2 进行镜头分割
        """
        if self.shot_detector is None:
            return self._fallback_windows(video_path)

        try:
            # TransNet 检测
            events = self.shot_detector.detect_shots(video_path, threshold=0.5)
            
            # 过滤短片段
            events = [e for e in events if (e[1] - e[0]) >= 0.5]
            
            # 优化：如果镜头太多，降采样防止 Graph 过大
            if len(events) > 800:
                events = events[::2] 

            if len(events) < 1: 
                return self._fallback_windows(video_path)
            return events

        except Exception as e:
            print(f"  ❌ TransNet execution failed ({e}), using fallback.")
            return self._fallback_windows(video_path)

    def _extract_event_features(self, video_path, events):
        # 既然没有 CUDA 冲突，我们尝试用 CPU 模式的 Decord 读取
        # (TransNet 用 GPU，这里读取用 CPU，互不干扰)
        if VideoReader is None: raise ImportError("decord required")
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except:
            return torch.tensor([]), torch.tensor([]), []
            
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        representative_frames = []
        
        for idx, (start_t, end_t) in enumerate(events):
            mid_t = (start_t + end_t) / 2.0
            frame_idx = int(mid_t * fps)
            if frame_idx >= total_frames: frame_idx = total_frames - 1
            
            try:
                frame_np = vr[frame_idx].asnumpy()
                representative_frames.append(Image.fromarray(frame_np))
            except:
                continue
        
        del vr # 释放资源
        
        if not representative_frames:
            return torch.tensor([]), torch.tensor([]), []

        # Batch Process
        batch_size = 64
        global_feats_list = []
        local_feats_list = []
        
        with torch.no_grad():
            for i in range(0, len(representative_frames), batch_size):
                batch = representative_frames[i : i+batch_size]
                inputs = self.clip_processor(images=batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # --- Global Features ---
                g_out = self.clip_model.get_image_features(**inputs)
                
                # 🔥 [修复报错]：如果是对象，提取 Tensor
                if isinstance(g_out, torch.Tensor):
                    g_feats = g_out
                elif hasattr(g_out, 'image_embeds'):
                    g_feats = g_out.image_embeds
                elif hasattr(g_out, 'pooler_output'):
                    g_feats = g_out.pooler_output
                else:
                    g_feats = g_out[0]
                
                # --- Local Features ---
                l_out = self.clip_model.vision_model(**inputs, output_hidden_states=True)
                
                # 🔥 [修复报错]：提取 Tensor
                if hasattr(l_out, 'last_hidden_state'):
                    l_feats = l_out.last_hidden_state[:, 1:, :] 
                elif isinstance(l_out, tuple):
                    l_feats = l_out[0][:, 1:, :]
                else:
                    l_feats = g_feats.unsqueeze(1) # 兜底

                global_feats_list.append(g_feats)
                local_feats_list.append(l_feats)
        
        if len(global_feats_list) == 0:
            return torch.tensor([]), torch.tensor([]), []

        global_feats = torch.cat(global_feats_list, dim=0)
        local_feats = torch.cat(local_feats_list, dim=0)
        
        return global_feats, local_feats, representative_frames

    def _construct_event_graph(self, global_feats, local_feats, events):
        # 节点多时使用 CPU 算图
        compute_device = self.device
        if global_feats.shape[0] > 600: 
            compute_device = torch.device('cpu')

        global_feats = global_feats.to(compute_device)
        local_feats = local_feats.to(compute_device)

        try:
            adj_semantic = compute_similarity_matrix(
                global_feats, local_feats, 
                tau=self.tau, 
                event_times=events, 
                threshold=self.delta
            )
        except RuntimeError:
            print("⚠️ Graph OOM, switching to CPU.")
            global_feats = global_feats.cpu()
            local_feats = local_feats.cpu()
            adj_semantic = compute_similarity_matrix(global_feats, local_feats, tau=self.tau, event_times=events, threshold=self.delta)
            
        Pi = compute_pagerank_matrix(adj_semantic, alpha=self.alpha)
        return Pi.to(self.device)

    def _select_subgraph(self, Pi, question, global_feats, events):
        inputs = self.clip_processor(text=[question], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            q_out = self.clip_model.get_text_features(**inputs)
            
            # 🔥 [修复报错]：文本特征也做同样检查
            if isinstance(q_out, torch.Tensor):
                q_feat = q_out
            elif hasattr(q_out, 'text_embeds'):
                q_feat = q_out.text_embeds
            elif hasattr(q_out, 'pooler_output'):
                q_feat = q_out.pooler_output
            else:
                q_feat = q_out[0]
            
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
        
        g_norm = global_feats / global_feats.norm(dim=-1, keepdim=True)
        relevance = torch.mm(g_norm, q_feat.t()).squeeze()
        if relevance.dim() == 0: relevance = relevance.unsqueeze(0)
        relevance = torch.clamp(relevance, min=0.0)
        
        costs = torch.full((len(events),), self.tokens_per_frame, device=self.device)
        
        selector = CELFSelector(Pi, relevance, costs, lambda_param=self.lambda_param)
        selected_indices = selector.select(budget=self.token_budget)
        
        return selected_indices

    def _fallback_windows(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration = count / fps if fps > 0 else 0
        except:
            duration = 0
            
        if duration == 0: return [(0.0, 1.0)]
        
        events = []
        step = 5.0
        if duration > 1800: step = 10.0
        
        for t in np.arange(0, duration, step):
            events.append((t, min(t + step, duration)))
            
        if not events: events = [(0.0, min(1.0, duration))]
        return events

    def _build_graph_cot_prompt(self, question, options, segments, adj_matrix, selected_indices):
        timeline_str = ""
        for idx, (start, end, original_idx) in enumerate(segments):
            timeline_str += f"- Node {idx+1} (Time: {start:.1f}s - {end:.1f}s): [Visual Content]\n"

        graph_hints = []
        for i in range(len(selected_indices)):
            for j in range(len(selected_indices)):
                if i == j: continue
                u, v = selected_indices[i], selected_indices[j]
                if adj_matrix[u, v] > 0.05: 
                    graph_hints.append(f"Node {i+1} is semantically related to Node {j+1}.")
        hints_str = "\n".join(graph_hints[:5])

        if isinstance(options, list):
            clean_opts = [str(o) for o in options]
            options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(clean_opts)])
        else:
            options_str = str(options)

        prompt = (
            f"You are analyzing a long video. I have selected key events for you based on a semantic graph.\n\n"
            f"User Query: {question}\n\n"
            f"Selected Key Events Timeline:\n{timeline_str}\n"
            f"Key Semantic Connections identified by the graph:\n{hints_str}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Instructions:\n"
            f"1. Analyze the visual content of each Node relevant to the query.\n"
            f"2. Connect the clues: If Node X and Node Y are related, combine their information.\n"
            f"3. Reason step-by-step to answer the query.\n"
            f"Answer:"
        )
        return prompt

    def process_and_inference(self, video_path, question, options):
        # 1. TransNet 检测 (GPU)
        events = self._detect_shot_boundaries(video_path)
        if not events: return "C"
        
        # 2. 提取特征 (含类型修复)
        global_feats, local_feats, frames = self._extract_event_features(video_path, events)
        if len(frames) == 0: return "C"

        # 3. 建图
        Pi = self._construct_event_graph(global_feats, local_feats, events)
        
        # 4. 选图
        sel_idx = self._select_subgraph(Pi, question, global_feats, events)
        if not sel_idx: sel_idx = [0]
        
        valid_sel_idx = [i for i in sel_idx if i < len(frames)]
        if not valid_sel_idx: valid_sel_idx = [0]
        valid_sel_idx = sorted(valid_sel_idx)

        # 限制帧数，防止 Context 溢出
        MAX_INFERENCE_FRAMES = 96
        if len(valid_sel_idx) > MAX_INFERENCE_FRAMES:
            indices = np.linspace(0, len(valid_sel_idx) - 1, MAX_INFERENCE_FRAMES).astype(int)
            valid_sel_idx = [valid_sel_idx[i] for i in indices]

        selected_frames = []
        for i in valid_sel_idx:
            img = frames[i]
            if self.target_size:
                img_resized = img.resize(self.target_size, Image.BICUBIC)
                selected_frames.append(img_resized)
            else:
                selected_frames.append(img)

        prompt = self._build_graph_cot_prompt(
            question, options, 
            [(events[i][0], events[i][1], i) for i in valid_sel_idx], 
            Pi, valid_sel_idx
        )
        prompt += "\nImportant: End your response with 'The answer is X.'"

        # 🔥 关键：改回 2048。40960 绝对会爆显存。
        return self.model.generate(
            selected_frames, 
            prompt, 
            options, 
            max_new_tokens=20480 
        )