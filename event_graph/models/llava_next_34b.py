import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from PIL import Image
import numpy as np
import re


class LLaVANext34BWrapper:
    def __init__(self, model_path="/llava-hf/LLaVA-NeXT-Video-34B-hf"):
        """        
        Args:
            model_path
        """
        print(f"[LLaVA-NeXT-34B] Loading model from {model_path}...")
        
        # Processor
        self.processor = LlavaNextVideoProcessor.from_pretrained(model_path)
        print(f"  ✓ Processor loaded")
        
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  
            device_map="auto",  
            low_cpu_mem_usage=True
        )
        self.model.eval()
        print(f"  ✓ Model loaded (FP16, device_map=auto)")
        
        self.device = next(self.model.parameters()).device
        print(f"  ✓ Device: {self.device}")
        
        self.recommended_frames = 32
        
    def generate(self, frames, question, options=None):
        num_frames = len(frames)
        if num_frames > self.recommended_frames:
            print(f"⚠️  Warning: {num_frames} frames provided, recommended max is {self.recommended_frames}")
        
        if options:
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            prompt = f"USER: <video>\n{question}\n{options_text}\nAnswer with the option's letter from the given choices directly.\nASSISTANT:"
        else:
            prompt = f"USER: <video>\n{question}\nASSISTANT:"
        
        # shape: (num_frames, H, W, 3)
        video_array = np.stack([np.array(f) for f in frames])
        
        inputs = self.processor(
            text=prompt,
            videos=video_array,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if 'pixel_values_videos' in inputs:
            inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(torch.float16)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        if "ASSISTANT:" in output_text:
            output_text = output_text.split("ASSISTANT:")[-1].strip()
        
        if options:
            answer = self._extract_option(output_text, len(options))
        else:
            answer = output_text
        
        return answer
    
    def _extract_option(self, text, num_options):
        text = text.strip().upper()
        
        patterns = [
            r'^([A-D])[.\s]',  # "A. " 或 "A "
            r'^([A-D])$',       # "A"
            r'ANSWER[:\s]*([A-D])',  # "ANSWER: A" or "ANSWER A"
            r'OPTION[:\s]*([A-D])',  # "OPTION: A"
            r'\(([A-D])\)',     # "(A)"
            r'([A-D])[.\s]',    #  "A. "
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                option = match.group(1)
                option_idx = ord(option) - ord('A')
                if 0 <= option_idx < num_options:
                    return option
        
        valid_options = [chr(65 + i) for i in range(num_options)]
        for char in text:
            if char in valid_options:
                return char
        
        print(f"⚠️  Warning: Could not extract option from: {text[:50]}..., defaulting to 'A'")
        return "A"


if __name__ == "__main__":
    print("=" * 80)
    print("Testing LLaVA-NeXT-34B Wrapper")
    print("=" * 80)
    
    test_frames = []
    for i in range(8):
        color = (255, 0, 0) if i < 4 else (0, 0, 255)
        img = Image.new('RGB', (336, 336), color=color)
        test_frames.append(img)
    
    wrapper = LLaVANext34BWrapper()
    
    question = "What colors appear in this video?"
    options = ["Red only", "Blue only", "Red and Blue", "Green and Yellow"]
    
    print(f"\n📝 Question: {question}")
    print(f"📋 Options: {options}")
    print(f"🎬 Frames: {len(test_frames)}")
    print("\n🔮 Generating...")
    
    answer = wrapper.generate(test_frames, question, options)
    
    print(f"\n✅ Answer: {answer}")
    print("=" * 80)
