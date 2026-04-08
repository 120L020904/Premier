import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import os
from typing import Optional, Callable, Tuple, Any
import matplotlib.pyplot as plt


class ParquetByteImageDataset(Dataset):
    """
    从Parquet文件中读取数据集，根据label_0的值选择对应的jpg图像

    数据格式要求:
    - label_0: 标签列，值为0或1
    - jpg_0: 当label_0=1时使用的图像字节数据
    - jpg_1: 当label_0=0时使用的图像字节数据

    图像数据必须是字节格式（bytes），使用指定的解码方式:
    Image.open(io.BytesIO(image_bytes)).convert("RGB")
    """

    def __init__(
            self,
            parquet_path: str,
            transform: Optional[Callable] = None,
            max_samples: Optional[int] = None,
            cache_dir: Optional[str] = None
    ):
        """
        初始化数据集

        Args:
            parquet_path: Parquet文件路径
            transform: 图像预处理转换
            max_samples: 最大样本数量（用于调试）
            cache_dir: 缓存目录（可选，用于加速重复加载）
        """
        self.parquet_path = parquet_path
        self.transform = transform

        # 加载数据
        print(f"正在加载Parquet文件: {parquet_path}")
        self.df = pd.read_parquet(parquet_path)

        # 验证必要列是否存在
        required_columns = ['label_0', 'jpg_0', 'jpg_1']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Parquet文件缺少必要列: {missing_cols}. "
                             f"可用列: {list(self.df.columns)}")

        # 验证数据类型
        self._validate_data_types()

        # 限制样本数量（用于调试）
        if max_samples is not None and max_samples < len(self.df):
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            print(f"限制样本数量为: {max_samples}")

        # 缓存处理
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"启用缓存目录: {cache_dir}")

        print(f"数据集加载完成，总样本数: {len(self.df)}")

    def _validate_data_types(self):
        """验证jpg_0和jpg_1列的数据类型是否为字节"""
        # 检查前几个样本的数据类型
        sample_size = min(5, len(self.df))
        for i in range(sample_size):
            row = self.df.iloc[i]

            if not isinstance(row['jpg_0'], (bytes, bytearray, np.ndarray)):
                raise TypeError(f"样本 {i} 的 jpg_0 列不是字节类型，类型为: {type(row['jpg_0'])}")

            if not isinstance(row['jpg_1'], (bytes, bytearray, np.ndarray)):
                raise TypeError(f"样本 {i} 的 jpg_1 列不是字节类型，类型为: {type(row['jpg_1'])}")

        print("✅ 数据类型验证通过: jpg_0 和 jpg_1 列均为字节类型")

    def _decode_image_bytes(self, image_bytes: Any) -> Image.Image:
        """
        使用指定方式解码图像字节数据

        Args:
            image_bytes: 图像字节数据（bytes, bytearray 或 numpy array）

        Returns:
            PIL.Image: 解码后的RGB图像
        """
        # 确保是字节类型
        if isinstance(image_bytes, np.ndarray):
            image_bytes = image_bytes.tobytes()
        elif isinstance(image_bytes, bytearray):
            image_bytes = bytes(image_bytes)
        elif not isinstance(image_bytes, bytes):
            raise TypeError(f"不支持的数据类型: {type(image_bytes)}。需要 bytes, bytearray 或 numpy array")

        # 使用指定的方式解码：Image.open(io.BytesIO(jpg_0)).convert("RGB")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        return image

    def _get_cached_image_path(self, idx: int, label_value: int) -> str:
        """获取缓存图像路径"""
        if not self.cache_dir:
            return None

        filename = f"sample_{idx}_label_{label_value}.jpg"
        return os.path.join(self.cache_dir, filename)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        获取单个样本的图像

        Args:
            idx: 样本索引

        Returns:
            image: 图像张量
        """
        row = self.df.iloc[idx]
        label_value = int(row['label_0'])

        # 选择对应的图像字节数据
        if label_value == 1:
            image_bytes = row['jpg_0']
        else:
            image_bytes = row['jpg_1']

        # 检查缓存
        cached_path = self._get_cached_image_path(idx, label_value)
        if cached_path and os.path.exists(cached_path):
            image = Image.open(cached_path).convert('RGB')
        else:
            # 使用指定的解码方式
            image = self._decode_image_bytes(image_bytes)
            # 保存到缓存
            if cached_path:
                image.save(cached_path, 'JPEG', quality=85)

        # 应用转换
        if self.transform:
            image = self.transform(image)

        return image

    def get_sample_info(self, idx: int) -> dict:
        """获取样本的详细信息（用于调试）"""
        row = self.df.iloc[idx]
        label_value = int(row['label_0'])

        return {
            'index': idx,
            'label_value': label_value,
            'selected_column': 'jpg_0' if label_value == 1 else 'jpg_1',
            'jpg_0_size': len(row['jpg_0']) if isinstance(row['jpg_0'], (bytes, bytearray)) else 'N/A',
            'jpg_1_size': len(row['jpg_1']) if isinstance(row['jpg_1'], (bytes, bytearray)) else 'N/A',
            'selected_size': len(row['jpg_0'] if label_value == 1 else row['jpg_1'])
        }

    def show_sample(self, idx: int):
        """显示样本图像"""
        # 获取原始图像用于显示（不应用transform）
        row = self.df.iloc[idx]
        label_value = int(row['label_0'])

        if label_value == 1:
            image_bytes = row['jpg_0']
        else:
            image_bytes = row['jpg_1']

        # 使用指定的解码方式
        image = self._decode_image_bytes(image_bytes)

        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(f'Sample {idx}, Label: {label_value}, Selected: {"jpg_0" if label_value == 1 else "jpg_1"}\n'
                  f'Image size: {image.size}, Mode: {image.mode}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    import torchvision.transforms as transforms

    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    dataset = ParquetByteImageDataset(
        parquet_path='your_data.parquet',  # 替换为你的Parquet文件路径
        transform=transform,
        max_samples=10,  # 用于调试
        cache_dir='./image_cache'
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # 调试时设为0，避免多进程问题
        pin_memory=True
    )

    # 测试数据集
    print(f"\n{'=' * 50}")
    print(f"数据集大小: {len(dataset)}")
    print(f"{'=' * 50}")

    # 显示几个样本
    for i in range(min(3, len(dataset))):
        print(f"\n样本 {i} 信息:")
        info = dataset.get_sample_info(i)
        for key, value in info.items():
            print(f"  {key}: {value}")

        print(f"显示样本 {i}...")
        dataset.show_sample(i)

    # 测试数据加载
    print(f"\n{'=' * 50}")
    print("测试数据加载器:")
    print(f"{'=' * 50}")

    for batch_idx, images in enumerate(dataloader):
        print(f"批次 {batch_idx}:")
        print(f"  图像形状: {images.shape}")
        print(f"  图像值范围: min={images.min().item():.3f}, max={images.max().item():.3f}")

        # 只测试一个批次
        break

    print(f"\n{'=' * 50}")
    print("✅ 所有测试通过！Dataset类工作正常")
    print(f"{'=' * 50}")