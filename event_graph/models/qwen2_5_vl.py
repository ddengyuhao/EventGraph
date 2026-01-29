import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

class Qwen2_5_VLWrapper:
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        print(f"🚀 [Qwen2.5-VL] Loading model from {model_path} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
        except Exception as e:
            print(f"⚠️ Flash Attention load failed, falling back to default: {e}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                device_map="auto"
            )

        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
        except:
            # Fallback if local path structure is weird
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            
        print("✅ Model loaded successfully.")

    def generate(self, video_frames, prompt, options=None, **kwargs):
        """
        Args:
            video_frames: List[PIL.Image] or similar
            prompt: str
            options: Unused here, but kept for interface compatibility
            **kwargs: Extra args like max_new_tokens
        """
        # 1. Construct Qwen-style messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_frames, # Qwen processor handles PIL list directly
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 2. Process Inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # 3. Inference
        # 🔥 关键修改：接收并使用 kwargs (如 max_new_tokens)
        # 默认 max_new_tokens 设为 1024 以防止截断，如果 kwargs 里有则覆盖
        gen_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": False, # Greedy decoding for determinism
            **kwargs 
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # 4. Decode
        trimmed_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]