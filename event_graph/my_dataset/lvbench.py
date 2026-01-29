import os
import json
import glob
from torch.utils.data import Dataset

class LVBenchDataset(Dataset):
    def __init__(self, root_dir="/root/icml2026/dataset/LVBench/LVBench", split="test", max_samples=None):
        """
        LVBench 数据集加载器 (适配用户截图目录结构)
        Args:
            root_dir: LVBench 项目的根目录 (包含 data, scripts 等文件夹)
            max_samples: 仅加载前 N 个样本用于快速测试
        """
        self.root_dir = root_dir
        self.samples = []
        
        print(f"📂 [LVBench] 初始化数据集，根目录: {self.root_dir}")

        # ==================================================
        # 1. 寻找元数据文件 (video_info.meta.jsonl)
        # 根据截图，它应该在 data/ 目录下
        # ==================================================
        meta_search_paths = [
            os.path.join(self.root_dir, "data", "*.jsonl"),      # 优先找 data/
            os.path.join(self.root_dir, "**", "*.jsonl")         # 备用：递归找
        ]
        
        meta_path = None
        for pattern in meta_search_paths:
            found = glob.glob(pattern, recursive=True)
            for f in found:
                if "meta" in os.path.basename(f): # 确保文件名包含 meta
                    meta_path = f
                    break
            if meta_path: break
        
        if not meta_path:
            raise FileNotFoundError(f"❌ 未找到 video_info.meta.jsonl！请检查 {self.root_dir}/data 目录。")
            
        print(f"📄 加载元数据: {meta_path}")

        # ==================================================
        # 2. 建立本地视频索引 (适配 scripts/tmp 和 scripts/videos)
        # ==================================================
        print("🔍 扫描本地视频文件...")
        video_map = {}
        
        # 定义搜索路径，根据你的截图：
        # 1. scripts/tmp/*.mp4 (你手动下载或缓存的)
        # 2. scripts/videos/**/*.mp4 (video2dataset 下载的)
        video_search_patterns = [
            os.path.join(self.root_dir, "scripts", "tmp", "*.mp4"),
            os.path.join(self.root_dir, "scripts", "videos", "**", "*.mp4"),
            os.path.join(self.root_dir, "**", "*.mp4") # 全局递归兜底
        ]
        
        found_videos_count = 0
        for pattern in video_search_patterns:
            files = glob.glob(pattern, recursive=True)
            for v_path in files:
                fname = os.path.basename(v_path)
                fid = os.path.splitext(fname)[0] # 获取文件名作为 ID (例如 2sriHX3PbXw)
                
                # 只有当该 ID 还没被记录时才添加 (避免重复)
                if fid not in video_map:
                    video_map[fid] = v_path
                    found_videos_count += 1

        print(f"   硬盘上共找到 {found_videos_count} 个视频文件。")

        # ==================================================
        # 3. 解析 JSONL 并构建样本
        # ==================================================
        skipped_count = 0
        
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    
                    entry = json.loads(line)
                    video_id = entry.get('key', '') # LVBench 使用 'key' (YouTube ID)
                    
                    # 关键步骤：检查该 ID 是否在我们的视频文件列表中
                    if video_id in video_map:
                        video_path = video_map[video_id]
                    else:
                        # 没下载视频则跳过
                        skipped_count += 1
                        continue 

                    # 遍历该视频下的所有问题 ('qa' 字段)
                    for q_item in entry.get('qa', []):
                        question = q_item.get('question', '')
                        
                        # 解析选项 (LVBench 通常是 A,B,C,D)
                        options = []
                        # 尝试从 option1...option4 字段读取
                        for opt_key in ['option1', 'option2', 'option3', 'option4']:
                            if opt_key in q_item:
                                options.append(q_item[opt_key])
                        
                        # 如果上面的方式没读到，尝试直接读取列表
                        if not options and 'options' in q_item:
                             options = q_item['options']

                        # 处理答案 (0->A, 1->B ...)
                        answer_raw = q_item.get('answer', '')
                        if isinstance(answer_raw, int):
                            answer = chr(65 + answer_raw) 
                        else:
                            answer = str(answer_raw).upper()

                        # 任务类型
                        task_type = q_item.get('question_type', 'general')

                        self.samples.append({
                            "id": f"{video_id}_{len(self.samples)}",
                            "video_path": video_path,
                            "question": question,
                            "options": options,
                            "answer": answer,
                            "task_type": task_type
                        })

        except Exception as e:
            print(f"❌ JSONL 读取失败: {e}")

        # 4. 截取小样本测试
        if max_samples is not None and max_samples > 0:
            print(f"✂️ [Test Mode] 截取前 {max_samples} 个样本进行测试。")
            self.samples = self.samples[:max_samples]

        print(f"✅ 数据集构建完成！")
        print(f"   - 匹配成功的视频数: {len(set(s['video_path'] for s in self.samples))}")
        print(f"   - 跳过(无视频): {skipped_count} 个条目")
        print(f"   - 最终题目数量: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]