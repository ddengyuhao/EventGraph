import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# 注意：如果是旧版 Qwen2-VL，请用 Qwen2VLForConditionalGeneration
# 但 Qwen2.5 的类通常向下兼容或者建议直接用 2.5 的权重
from qwen_vl_utils import process_vision_info
from PIL import Image

class Qwen2_VL_72B_Wrapper:
    def __init__(self, model_path="/root/hhq/models/Qwen2.5-VL-72B-Instruct"):
        """
        72B 模型包装器
        Args:
            model_path: 72B 模型的本地路径
        """
        print(f"🚀 [Qwen2.5-VL-72B] Loading model from {model_path} ...")
        
        # 显存预警
        if torch.cuda.device_count() < 2:
            print("⚠️ Warning: 72B model typically requires 2+ A100s or 4+ Consumer GPUs.")
        
        # 1. 加载模型 (核心差异：device_map="auto")
        # 这会自动将模型层切分到 GPU 0, 1, 2, 3...
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"  # <--- 🔥 关键：自动多卡并行
            )
        except Exception as e:
            print(f"⚠️ Load failed, falling back to float16: {e}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                device_map="auto"
            )

        # 2. 加载 Processor
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
        # 输入数据移动到模型所在的第一块 GPU（accelerate 会自动处理剩下的传播）
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