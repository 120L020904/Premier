import os
import json
from pathlib import Path

import torch
from diffusers import ZImagePipeline

# 假设你的 jsonl 文件都在 data_dir 目录下
data_dir = "/hdd5/wangzihao/data/PIP-dataset"
output_dir = "/hdd5/wangzihao/data/PIP-dataset-image-98"
# 确保输出目录存在
Path(output_dir).mkdir(parents=True, exist_ok=True)

# 获取所有 .jsonl 文件，取前50个（按文件名排序以确保确定性）
# jsonl_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jsonl')])[:25]
jsonl_files = ["98.jsonl"]
pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", dtype=torch.bfloat16, local_files_only=True)
pipe = pipe.to("cuda:6")  # 使用GPU加速
# 假设 z-image 模型的调用函数如下（你需要替换成实际调用方式）


for filename in jsonl_files:
    file_path = os.path.join(data_dir, filename)
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                continue
            prompt = data['prompt']
            img_id = data['id']

            # 调用模型生成图像
            image = pipe(prompt=prompt, num_inference_steps=20, height=512, width=512, guidance_scale=0.0).images[0]

            # 保存图像，使用 id 作为文件名
            output_path = os.path.join(output_dir, f"{img_id}.png")
            image.save(output_path)