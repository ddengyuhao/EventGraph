import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from .base_method import BaseMethod
from .graph_builder import compute_similarity_matrix, compute_pagerank_matrix
from .celf_solver import CELFSelector

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None

class EventGraphLMM(BaseMethod):
    def __init__(self, args, model):
        super().__init__(args, model)
        
        # Args
        self.tau = getattr(args, 'tau', 30.0)
        self.delta = getattr(args, 'delta', 0.65)
        self.alpha = getattr(args, 'alpha', 0.15)
        self.token_budget = getattr(args, 'token_budget', 8192)
        
        # Config Tokens
        bb = getattr(args, 'backbone', '')
        if 'Qwen' in bb:
            self.tokens_per_frame = 256
            self.target_size = (336, 336)
        else:
            self.tokens_per_frame = 576 # Default for LLaVA-Next etc.
            self.target_size = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load CLIP
        self._load_clip(args)
        
        # 2. Load Detector (Optional)
        self.shot_detector = None
        try:
            from .transnet_detector import TransNetV2Detector
            # Allow clean fallback if GPU is busy or incompatible
            self.shot_detector = TransNetV2Detector(device='cuda' if torch.cuda.is_available() else 'cpu')
            print(f"✅ TransNetV2 loaded on {self.shot_detector.device}")
        except ImportError:
            print("⚠️ TransNetV2 not found. Using fallback windows.")
        except Exception as e:
            print(f"⚠️ TransNetV2 init failed: {e}. Using fallback.")

    def _load_clip(self, args):
        path = getattr(args, 'clip_path', None) or "openai/clip-vit-large-patch14"
        print(f"📥 Loading CLIP from: {path}")
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(path)
            self.clip_model = CLIPModel.from_pretrained(path).to(self.device).eval()
        except Exception as e:
            print(f"❌ Failed to load CLIP: {e}. Trying default OpenAI.")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device).eval()

    def _detect_shot_boundaries(self, video_path):
        """Detects shot boundaries using TransNetV2 or fallback."""
        if self.shot_detector is None:
            return self._fallback_windows(video_path)

        try:
            events = self.shot_detector.detect_shots(video_path, threshold=0.5)
            # Filter short noise < 0.5s
            events = [e for e in events if (e[1] - e[0]) >= 0.5]
            
            # Downsample if too many events (Optimization for very long videos)
            if len(events) > 800:
                events = events[::2] 

            if len(events) < 1: 
                return self._fallback_windows(video_path)
            return events

        except Exception as e:
            print(f"❌ [EventGraph] Shot detection failed ({e}). Using fallback.")
            return self._fallback_windows(video_path)

    def _extract_event_features(self, video_path, events):
        """Extracts visual features for each event using CLIP."""
        if VideoReader is None: 
            raise ImportError("decord is required for video reading.")
        
        try:
            # Use CPU for video decoding to avoid potential CUDA conflicts with TransNet
            vr = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            print(f"❌ [EventGraph] Video read failed: {e}")
            return torch.tensor([]), torch.tensor([]), []
            
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        representative_frames = []
        
        for idx, (start_t, end_t) in enumerate(events):
            mid_t = (start_t + end_t) / 2.0
            frame_idx = int(mid_t * fps)
            frame_idx = min(frame_idx, total_frames - 1)
            
            try:
                frame_np = vr[frame_idx].asnumpy()
                representative_frames.append(Image.fromarray(frame_np))
            except:
                continue
        
        del vr # Explicitly release decord resources
        
        if not representative_frames:
            return torch.tensor([]), torch.tensor([]), []

        # Batch Feature Extraction
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
                g_feats = self._safe_extract_tensor(g_out)
                
                # --- Local Features ---
                l_out = self.clip_model.vision_model(**inputs, output_hidden_states=True)
                if hasattr(l_out, 'last_hidden_state'):
                    l_feats = l_out.last_hidden_state[:, 1:, :] 
                elif isinstance(l_out, tuple):
                    l_feats = l_out[0][:, 1:, :]
                else:
                    l_feats = g_feats.unsqueeze(1) # Fallback

                global_feats_list.append(g_feats)
                local_feats_list.append(l_feats)
        
        if len(global_feats_list) == 0:
            return torch.tensor([]), torch.tensor([]), []

        global_feats = torch.cat(global_feats_list, dim=0)
        local_feats = torch.cat(local_feats_list, dim=0)
        
        return global_feats, local_feats, representative_frames

    def _safe_extract_tensor(self, output):
        """Helper to extract tensor from various HuggingFace output formats."""
        if isinstance(output, torch.Tensor):
            return output
        elif hasattr(output, 'image_embeds'):
            return output.image_embeds
        elif hasattr(output, 'text_embeds'):
            return output.text_embeds
        elif hasattr(output, 'pooler_output'):
            return output.pooler_output
        elif isinstance(output, (list, tuple)):
            return output[0]
        return output

    def _construct_event_graph(self, global_feats, local_feats, events):
        """Constructs the semantic-temporal graph."""
        # Use CPU if graph is too large to fit in GPU memory
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
            print("⚠️ [EventGraph] Graph construction OOM, switching to CPU.")
            global_feats = global_feats.cpu()
            local_feats = local_feats.cpu()
            adj_semantic = compute_similarity_matrix(
                global_feats, local_feats, 
                tau=self.tau, 
                event_times=events, 
                threshold=self.delta
            )
            
        Pi = compute_pagerank_matrix(adj_semantic, alpha=self.alpha)
        return Pi.to(self.device)

    def _select_subgraph(self, Pi, question, global_feats, events):
        """Selects key events using CELF algorithm based on query relevance."""
        inputs = self.clip_processor(text=[question], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            q_out = self.clip_model.get_text_features(**inputs)
            q_feat = self._safe_extract_tensor(q_out)
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
        """Fallback method using uniform temporal windows."""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration = count / fps if fps > 0 else 0
        except:
            duration = 0
            
        if duration == 0: return [(0.0, 1.0)]
        
        # Adaptive step size based on video length
        step = 5.0
        if duration > 1800: step = 10.0
        
        events = []
        for t in np.arange(0, duration, step):
            events.append((t, min(t + step, duration)))
            
        if not events: events = [(0.0, min(1.0, duration))]
        return events

    def _build_graph_cot_prompt(self, question, options, segments, adj_matrix, selected_indices):
        """Builds the Chain-of-Thought prompt with graph insights."""
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
        # 1. Shot Detection (TransNet V2 on GPU)
        events = self._detect_shot_boundaries(video_path)
        if not events: return "C"
        
        # 2. Feature Extraction
        global_feats, local_feats, frames = self._extract_event_features(video_path, events)
        if len(frames) == 0: return "C"

        # 3. Graph Construction
        Pi = self._construct_event_graph(global_feats, local_feats, events)
        
        # 4. Keyframe Selection (CELF)
        sel_idx = self._select_subgraph(Pi, question, global_feats, events)
        if not sel_idx: sel_idx = [0]
        
        # Filter indices ensuring they exist in extracted frames
        valid_sel_idx = [i for i in sel_idx if i < len(frames)]
        if not valid_sel_idx: valid_sel_idx = [0]
        valid_sel_idx = sorted(valid_sel_idx)

        # 5. Prepare Input Frames (Limit frame count to avoid OOM)
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

        # 6. Build Prompt
        prompt = self._build_graph_cot_prompt(
            question, options, 
            [(events[i][0], events[i][1], i) for i in valid_sel_idx], 
            Pi, valid_sel_idx
        )
        prompt += "\nImportant: End your response with 'The answer is X.'"

        # 7. Model Generation
        return self.model.generate(
            selected_frames, 
            prompt, 
            options, 
            max_new_tokens=4096 
        )