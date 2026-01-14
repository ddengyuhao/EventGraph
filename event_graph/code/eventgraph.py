# /root/hhq/main_code/methods/eventgraph.py
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from .base_method import BaseMethod
from .graph_builder import compute_similarity_matrix, compute_pagerank_matrix
from .celf_solver import CELFSelector
from .uboco_detector import UbocoDetector

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None

class EventGraphLMM(BaseMethod):
    def __init__(self, args, model):
        super().__init__(args, model)
        
        # Params from Paper Section 4.1 [cite: 665, 666, 667]
        self.tau = 30.0  
        self.delta = 0.65 
        self.alpha = 0.15 
        self.lambda_param = 1.0 
        self.token_budget = args.token_budget
        
        # Detect token density for budget calculation
        self.tokens_per_frame = 576 if '34B' in getattr(args, 'backbone', '') else 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load CLIP Once
        self._load_clip_model()
        
        # 2. Initialize Uboco with the SHARED model
        self.shot_detector = UbocoDetector(
            device=self.device, 
            clip_model=self.clip_model, 
            clip_processor=self.clip_processor
        )

    def _load_clip_model(self):
        # Uses local path if available to save download time
        local_path = "/root/hhq/models/clip-vit-large-patch14"
        model_name = local_path if torch.cuda.is_available() else "openai/clip-vit-large-patch14"
        
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.clip_model.eval()
        except Exception as e:
            print(f"Warning: Loading CLIP from {model_name} failed, trying openai default.")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.clip_model.eval()

    def _detect_shot_boundaries(self, video_path):
        # Delegate to the shared-model Uboco detector
        # Maps to Paper Section 3.2 "Event Nodes" [cite: 353]
        try:
            boundaries = self.shot_detector.detect(video_path, sample_rate=2) # sample_rate=2 for speed
            events = self._boundaries_to_events(boundaries, video_path)
            # Filter noise < 1s
            events = [e for e in events if (e[1] - e[0]) >= 1.0]
            if len(events) < 3: return self._fallback_windows(video_path)
            return events
        except Exception as e:
            print(f"  [EventGraph] Uboco failed ({e}), using fallback.")
            return self._fallback_windows(video_path)

    def _extract_event_features(self, video_path, events):
        # Optimization: Batch processing for CLIP instead of loop
        if VideoReader is None: raise ImportError("decord required")
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        
        representative_frames = []
        for start_t, end_t in events:
            mid_t = (start_t + end_t) / 2.0
            frame_idx = min(len(vr) - 1, int(mid_t * fps))
            frame_np = vr[frame_idx].asnumpy()
            representative_frames.append(Image.fromarray(frame_np))
        
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
                
        global_feats = torch.cat(global_feats_list, dim=0)
        local_feats = torch.cat(local_feats_list, dim=0)
        
        return global_feats, local_feats, representative_frames

    # ... [Keep _construct_event_graph, _select_subgraph, _boundaries_to_events, _fallback_windows as you wrote them] ...
    
    def _fallback_windows(self, video_path):
        # Helper for fallback
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = count / fps if fps > 0 else 0
        events = []
        for t in np.arange(0, duration, 2.0):
            events.append((t, min(t + 2.0, duration)))
        return events

    def _build_graph_cot_prompt(self, question, options, segments, adj_matrix, selected_indices):
        # Paper Section 3.4 [cite: 638]
        # Keep your simplified prompt, it is good for LLaVA-7B context limits
        event_timeline = [f"Event{i+1}" for i, _, _ in segments]
        prompt = f"Question: {question}\nOptions: {options}\nEvents: {'->'.join(event_timeline)}\nBased on these visual events, reason step-by-step and choose the best answer."
        return prompt

    def process_and_inference(self, video_path, question, options):
        # ... [Your existing logic is correct] ...
        events = self._detect_shot_boundaries(video_path)
        if not events: return "C"
        
        global_feats, local_feats, frames = self._extract_event_features(video_path, events)
        adj = self._construct_event_graph(global_feats, local_feats, events)
        sel_idx = self._select_subgraph(adj, question, global_feats, events)
        
        # Prepare selected frames
        selected_frames = [frames[i] for i in sorted(sel_idx)]
        
        # Inference
        prompt = self._build_graph_cot_prompt(question, options, 
                                            [(events[i][0], events[i][1], i) for i in sel_idx], 
                                            adj, sel_idx)
        
        return self.model.generate(selected_frames, prompt, options)