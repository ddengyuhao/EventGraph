import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

class Qwen2_VL_72B_Wrapper:
    def __init__(self, model_path="/root/hhq/models/Qwen2.5-VL-72B-Instruct"):
        """
        72B 
        Args:
            model_path
        """
        print(f"🚀 [Qwen2.5-VL-72B] Loading model from {model_path} ...")
        
        if torch.cuda.device_count() < 2:
            print("⚠️ Warning: 72B model typically requires 2+ A100s or 4+ Consumer GPUs.")
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"  
            )
        except Exception as e:
            print(f"⚠️ Load failed, falling back to float16: {e}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                device_map="auto"
            )

        self.processor = AutoProcessor.from_pretrained(model_path)
        print("✅ Model loaded successfully across GPUs.")

    def generate(self, video_frames, prompt, options=None, max_new_tokens=1024, **kwargs):
        # 1. Construct Messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_frames,
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
        
        # 3. Prepare Tensors
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # 4. Inference
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            **kwargs 
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # 5. Decode
        trimmed_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]