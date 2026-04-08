import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as T


class ImagePromptDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, transform=T.ToTensor(), target_size=(512, 512), drop_text_prob=0.1):
        """
        Args:
            jsonl_path (str): 路径到 .jsonl 文件。
            image_dir (str): 图像文件所在的目录。
            transform (callable, optional): 可选的图像变换。
        """
        self.image_dir = image_dir
        self.transform = transform
        self.data = []
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_id = entry['id']
        prompt = entry['prompt']

        # 假设图像格式为 .jpg 或 .png，可根据实际情况调整
        image_path = os.path.join(self.image_dir, image_id)
        if not os.path.splitext(image_path)[1]:
            # 如果 id 不含扩展名，尝试添加常见扩展名
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = image_path + ext
                if os.path.exists(candidate):
                    image_path = candidate
                    break

        image = Image.open(image_path).convert('RGB').resize(self.target_size)

        if self.transform:
            image = self.transform(image)
        if random.random() < self.drop_text_prob:
            prompt = ""
        return {
            'image': image,
            "description": prompt,
            'idx': image_id
        }
