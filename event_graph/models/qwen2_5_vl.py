# models/qwen2_5_vl.py
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

class Qwen2_5_VLWrapper:
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Qwen2.5-VL from {model_path}...")

        # --- 修改部分开始 ---
        # 使用 "sdpa" (PyTorch Native Scaled Dot Product Attention)
        # 它在 A100 上会自动启用 Flash Attention 后端，无需额外安装库
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",  # <--- 改这里！不要用 flash_attention_2
            device_map="auto"
        )
        # --- 修改部分结束 ---

        self.processor = AutoProcessor.from_pretrained(model_path)
        print("Qwen2.5-VL loaded successfully.")

    def generate(self, frames, prompt, options=None):
        """
        Args:
            frames: List[PIL.Image] - EventGraph 选出的关键帧
            prompt: str - 也就是 _build_graph_cot_prompt 生成的文本
            options: List[str] - 选项列表 (A, B, C, D...)
        """
        # 1. 构造 Messages
        # Qwen2.5-VL 接收 image 和 text 的混合输入
        content = []
        for img in frames:
            content.append({"type": "image", "image": img})
        
        # 将提示词加入
        content.append({"type": "text", "text": prompt})
        
        # 如果有选项，也可以在这里进一步格式化，不过你的 prompt 已经包含了 options
        # 这里为了引导模型输出答案，可以添加一个 "Answer:" 的引导
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        # 2. 预处理
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

        # 3. 推理
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,  # 不需要太长，只需输出推理和选项
                temperature=0.1,     # 低温采样，保证稳定性
                top_p=0.9
            )
        
        # 4. 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text