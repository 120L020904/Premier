import ctypes
import math
import os
import random
import subprocess

import pandas as pd
import torch
import yaml
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPProcessor

from torch.nn import functional as F
import torch
import torch.distributed as dist
from torch.autograd import Function

def save_images(pil_images,
                folder_path,
                case_number,
                seed
                ):
    for num, im in enumerate(pil_images):
        os.makedirs(f"{folder_path}", exist_ok=True)
        im.save(f"{folder_path}/{case_number}_{num + seed}.png")
    return


def get_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get("OMINI_CONFIG")
    assert config_path is not None, "Please set the OMINI_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(config_path)
    return config

def _encode_prompt_with_clip(
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        text_input_ids=None,
        num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def clip_cos(prompt_1, prompt_2, clip_text_encoder, tokenizer_clip):
    prompt_embeds_1 = _encode_prompt_with_clip(clip_text_encoder, tokenizer_clip, prompt_1, clip_text_encoder.device, )
    prompt_embeds_2 = _encode_prompt_with_clip(clip_text_encoder, tokenizer_clip, prompt_2, clip_text_encoder.device, )
    cos_sim = torch.cosine_similarity(prompt_embeds_1.view(-1), prompt_embeds_2.view(-1), dim=-1)
    return cos_sim


def get_image_dxm(item_list, image_dir):
    image_list = []
    bad_image_list = []
    for item in item_list:
        positive_image_name = item[1]
        negative_image_name = item[0]
        positive_image_path = os.path.join(image_dir, positive_image_name)
        negative_image_path = os.path.join(image_dir, negative_image_name)
        image = Image.open(positive_image_path)
        negative_image = Image.open(negative_image_path)
        bad_image_list.append(negative_image)
        image_list.append(image)
    return image_list, bad_image_list

def get_image_dxm_release(item_list, image_dir):
    image_list = []
    
    for item in item_list:
        positive_image_name = item[0]
        positive_image_path = os.path.join(image_dir, positive_image_name)
        image = Image.open(positive_image_path)
        image_list.append(image)
    return image_list

def get_ref(input_prompt, image_data, top=4, clip_model=None):
    # csv_path = "/data/wangzihao/DiffusionDPO/user_csv/1013.csv"

    sim_score_list = []
    for item in image_data:
        ref_prompt = item[2]
        sim_score = clip_cos(input_prompt, ref_prompt, clip_model)
        row_item = {"item": item, "sim_score": sim_score}
        sim_score_list.append(row_item)
    sim_score_list.sort(key=lambda x: x['sim_score'], reverse=True)
    ref_list = sim_score_list[1:top + 1]
    return ref_list, sim_score_list[0]["sim_score"]


def get_ref_csv(input_prompt, csv_path, is_dxm=True, top=4, clip_model=None):
    # csv_path = "/data/wangzihao/DiffusionDPO/user_csv/1013.csv"
    if is_dxm:
        df = pd.read_csv(csv_path, index_col=0)
    else:
        df = pd.read_csv(csv_path)
    grouped = df.groupby('caption', as_index=False)

    sim_score_list = []
    for name, group in grouped:
        sim_score = clip_cos(input_prompt, name, clip_model)
        row_item = {"group": group, "sim_score": sim_score, "name": name}
        sim_score_list.append(row_item)
    sim_score_list.sort(key=lambda x: x['sim_score'], reverse=True)
    ref_list = []
    for i in range(top):
        ref_row = sim_score_list[i]["group"].sample(n=1)
        ref_list.append(ref_row)
    return ref_list, sim_score_list[0]["sim_score"], sim_score_list[0]["name"]


def get_image_dxm_csv(df_list, image_dir):
    image_list = []
    for df in df_list:

        if isinstance(df, pd.DataFrame):
            index = df.index[random.randint(0, len(df) - 1)]
            row = df.loc[index]
        else:
            row = df
        positive_image_name = row["positive_image"]
        positive_image_path = os.path.join(image_dir, positive_image_name)
        image = Image.open(positive_image_path)
        image_list.append(image)
    return image_list


def get_related_images(input_row, csv_path, num=5):
    df = pd.read_csv(csv_path, index_col=0)
    grouped = df.groupby('caption', as_index=False)
    input_prompt = input_row["caption"]
    sim_score_list = []
    for name, group in grouped:
        sim_score = clip_cos(input_prompt, name)
        row_item = {"group": group, "sim_score": sim_score}
        sim_score_list.append(row_item)
    sim_score_list.sort(key=lambda x: x['sim_score'], reverse=True)
    k = 0
    image_id_set = set()
    row_list = []
    for group_item in sim_score_list:
        for index, row in group_item["group"]:
            if row["best_image_uid"] == input_row["best_image_uid"] or row["best_image_uid"] in image_id_set:
                continue
            else:
                image_id_set.add(row["best_image_uid"])
                row_list.append(row)
                k += 1
            if k >= num:
                break
    return row_list


def clip_img_vector(input_image, clip_model, preprocess):
    image = preprocess(images=input_image, return_tensors="pt")["pixel_values"].to(clip_model.device)
    with torch.no_grad():
        embedding = clip_model.get_image_features(image)
    return embedding


def find_latest_checkpoint(checkpoint_dir, is_number=True):
    """自动查找最新的检查点文件"""

    def is_number_string(s):
        """检查字符串是否为整型、浮点型或科学计数法数字"""
        try:
            float(s)  # 尝试转换为浮点数
            return True
        except ValueError:
            return False

    if not os.path.exists(checkpoint_dir):
        return None

    # 获取所有检查点文件
    if is_number:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir)
                            if is_number_string(f)]
    else:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir)]

    # 按修改时间排序
    if checkpoint_files:
        # 获取完整路径并按修改时间排序
        checkpoint_paths = [os.path.join(checkpoint_dir, f)
                            for f in checkpoint_files]
        checkpoint_paths.sort(key=os.path.getmtime)
        return checkpoint_paths[-1]  # 返回最新的文件
    return None


def pad_to_square(pil_image):
    new_size = max(pil_image.width, pil_image.height)
    square_image = Image.new("RGB", (new_size, new_size), "white")
    left = (new_size - pil_image.width) // 2
    top = (new_size - pil_image.height) // 2
    square_image.paste(pil_image, (left, top))
    return square_image


def pad_to_target(pil_image, target_size):
    original_width, original_height = pil_image.size
    target_width, target_height = target_size

    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # Pad the image to the target aspect ratio
    if original_aspect_ratio > target_aspect_ratio:
        new_width = original_width
        new_height = int(new_width / target_aspect_ratio)
    else:
        new_height = original_height
        new_width = int(new_height * target_aspect_ratio)

    pad_image = Image.new("RGB", (new_width, new_height), "white")
    left = (new_width - original_width) // 2
    top = (new_height - original_height) // 2
    pad_image.paste(pil_image, (left, top))

    # Resize the image to the target size
    resized_image = pad_image.resize(target_size)
    return resized_image


def load_clip(pipeline, config, torch_dtype, device, ckpt_dir=None, is_training=False):
    model_path = os.getenv("CLIP_MODEL_PATH", "openai/clip-vit-large-patch14")
    clip_model = CLIPVisionModelWithProjection.from_pretrained(model_path).to(device, dtype=torch_dtype)
    clip_processor = CLIPProcessor.from_pretrained(model_path)
    pipeline.pipe.clip_model = clip_model
    pipeline.pipe.clip_processor = clip_processor


def tokenize_t5_prompt(pipe, input_prompt, max_length, **kargs):
    return pipe.tokenizer_2(
        input_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
        **kargs,
    )


def unpad_input_ids(input_ids, attention_mask):
    return [input_ids[i][attention_mask[i].bool()][:-1] for i in range(input_ids.shape[0])]


def get_embedding_path(model_path):
    run_name_list = os.listdir(model_path)
    user_tensor_dict = {}
    for run_name in run_name_list:

        path = os.path.join(model_path, run_name, "ckpt/500")
        if os.path.isdir(path):
            file_list = os.listdir(path)
            tensor_file = [
                f for f in file_list
                if os.path.isfile(os.path.join(path, f)) and f.endswith("safetensors")
            ]
            user_id = tensor_file[0].split(".")[0].split("_")[-1]
            user_tensor_dict[user_id] = os.path.join(path, tensor_file[0])
    return user_tensor_dict


def negative_pair_loss(z: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    仅负样本对参与的分散损失 (基于负余弦相似度)
    Args:
        z: 输入特征 [B, ..., C]，自动展平为 [B, D]
        tau: 温度系数（默认0.1）
    Returns:
        loss: 标量损失值
    """
    B = z.shape[0]
    z_flat = z.reshape(B, -1)  # [B, D]

    # 1. 计算全量余弦相似度矩阵 [B, B]
    z_norm = F.normalize(z_flat, p=2, dim=-1)  # L2归一化
    cos_sim = z_norm @ z_norm.T  # [B, B]

    # 2. 生成负样本对掩码（排除自身对）
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)  # 非对角线位置为True
    neg_cos_sim = -cos_sim[mask].view(B, B - 1)  # 提取负样本相似度 [B, B-1]

    # 3. 计算指数项并求平均 (含数值稳定技巧)
    logits = -neg_cos_sim / tau  # 等价于 e^{sim(z_i,z_j)/τ}
    logits_max, _ = logits.max(dim=1, keepdim=True)
    exp_terms = torch.exp(logits - logits_max.detach())  # 防止指数溢出
    mean_exp = exp_terms.mean(dim=1)  # 每个样本的负对平均相似度

    # 4. 损失计算 (log-sum-exp形式)
    loss = (logits_max.squeeze() + torch.log(mean_exp)).mean()
    return loss


def stable_disp_loss(z: torch.Tensor, tau=1.0) -> torch.Tensor:
    """
    数值稳定的 Dispersive Loss (InfoNCE-L2 变体)
    Args:
        z: 输入特征 [B, ..., C] -> 自动展平为 [B, D]
        tau: 温度系数，控制分散强度（默认0.5）
    Returns:
        loss: 标量损失值
    """
    B = z.shape[0]
    z_flat = z.reshape(B, -1)  # [B, D]

    # 1. 计算全量距离矩阵 [B, B]
    dist_matrix = torch.cdist(z_flat.to(dtype=torch.float32), z_flat.to(dtype=torch.float32), p=2)  # [B, B]

    # 2. 排除自身距离（对角线置零）
    mask = torch.eye(B, dtype=torch.bool, device=z.device)
    dist_matrix = dist_matrix.masked_fill(mask, 0)  # 自身距离置零

    # 3. 提取有效距离（非对角线元素）
    valid_dist = dist_matrix[~mask].view(B, B - 1)  # [B, B-1]

    # 4. 数值稳定计算：log(mean(exp(-D/tau)))
    # 步骤分解：
    #   a) 计算指数项输入：-valid_dist / tau
    #   b) 用 logsumexp 替代 log(mean(exp(x))) = logsumexp(x) - log(n)
    n_valid = B-1  # 有效样本对数量
    logits = -valid_dist / tau  # [B, B-1]

    # Log-Sum-Exp 稳定计算
    if B > 1:
        logsumexp_per_row = torch.logsumexp(logits, dim=1)
        log_mean_exp = logsumexp_per_row - torch.log(
            torch.tensor(n_valid, device=z.device))
        return log_mean_exp.mean()  # 批次平均
    else:
        return torch.tensor(0, device=z.device, dtype=z.dtype)
def disp_loss(z: torch.Tensor, tau: float = 1) -> torch.Tensor:
    """
    Dispersive Loss (InfoNCE-L2 变体)
    Args:
        z: 输入特征 [B, ..., C] -> 自动展平为 [B, D]
        tau: 温度系数，控制分布集中度（默认0.5）
    Returns:
        loss: 标量损失值
    """
    B = z.shape[0]
    if B > 1:
        z_flat = z.reshape(B, -1)  # [B, D]

        # 计算所有样本对间的平方L2距离 [B, B]
        dist_matrix = torch.cdist(z_flat.to(dtype=torch.float32), z_flat.to(dtype=torch.float32), p=2).pow(2)  # [B, B]

        # 排除自身距离（对角线置零）
        mask = torch.eye(B, dtype=torch.bool, device=z.device)
        dist_matrix = dist_matrix.masked_fill(mask, 0)  # 自身距离不参与计算

        # 计算损失：log(mean(exp(-D/tau)))
        exp_terms = torch.exp(-dist_matrix / tau)  # 指数项 [B, B]
        exp_terms = exp_terms.masked_fill(mask, 0)  # 排除对角线
        loss = torch.log(exp_terms.sum() / (B * (B - 1)) + 1e-10)  # 仅非对角线对参与平均
    else:
        return torch.tensor(0, device=z.device, dtype=z.dtype)
    # print(f"loss {loss.to(dtype=torch.bfloat16)}")
    return loss.to(dtype=torch.bfloat16)


class AllGather(Function):
    """
    一个自定义的可微 all_gather 操作，确保梯度能正确传播。
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        # 创建一个用于存放所有进程数据的列表
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        # 执行 all_gather
        dist.all_gather(output, tensor)
        # 将列表拼接成一个大的张量
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # 在反向传播中，我们需要将全局梯度正确地分配回每个进程。
        # grad_output 是对应于 forward 输出（全局张量）的梯度。

        # 获取 world_size 和 rank
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # 将全局梯度按照 world_size 切分
        # 每个进程只关心对应自己那一部分的梯度
        chunk_size = grad_output.shape[0] // world_size
        grad_input = grad_output.narrow(0, rank * chunk_size, chunk_size)

        # 尽管每个进程只取了自己的一部分梯度，但为了与DDP的梯度平均机制保持一致，
        # 这里的梯度实际上需要是所有进程上梯度的总和。
        # 因此，我们需要对梯度执行 all_reduce (sum)。
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM)

        return grad_input


def distributed_disp_loss(all_gather,z_local: torch.Tensor, tau: float = 1.0, ) -> torch.Tensor:
    """
    分布式的 Dispersive Loss 实现。

    Args:
        z_local: 当前 GPU 上的特征张量 [local_B, ..., C]
        tau: 温度系数
    Returns:
        loss: 标量损失值
    """
    # 检查是否处于分布式环境中
    if not dist.is_available() or not dist.is_initialized():
        # 如果不是分布式环境，直接计算本地损失
        return stable_disp_loss(z_local, tau)

    # 1. 从所有 GPU 收集特征
    # z_local: [local_B, C]
    # z_global: [local_B * world_size, C]
    z_global = all_gather(z_local)

    # 2. 在收集到的全局特征上计算损失
    # 现在 z_global 包含了整个 global batch 的所有样本
    loss = stable_disp_loss(z_global, tau)

    return loss
def is_image_corrupted(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证文件完整性
        return False  # 未抛出异常则文件正常
    except Exception as e:
        print(f"损坏文件: {file_path} - 错误: {e}")
        return True
def generate_virtual_user_preferences(attributes_json):
    """
    根据图像属性JSON生成虚拟用户偏好

    Args:
        attributes_json (dict): 解析后的图像属性JSON对象
        preference_strategy (str): 偏好生成策略，例如 'random'（随机）

    Returns:
        dict: 虚拟用户的所有偏好属性
    """
    user_preferences = [] # 初始化用户偏好字典

    # 遍历所有一级属性
    for level1_key, level1_value in attributes_json.items():
        # 判断一级属性的值类型：是字典（包含二级属性）还是列表/值（直接属性）
        if isinstance(level1_value, dict):
            # 处理有二级属性的情况（如fruits, animals）
            # 1. 决定用户是否偏好这个一级类别下的内容（这里假设总是偏好）
            # 2. 遍历该一级类别下的所有二级属性
            preferred_subitems = {}
            select_key = random.sample(list(level1_value.keys()),1)[0]
            level2_value = level1_value[select_key]
            select_attribute = random.sample(list(level2_value), 1)[0]
            # 如果用户偏好该一级类别下的至少一个子项，则将其添加到用户偏好中
            user_preferences.append(f"{level1_key} is {select_key} {select_attribute}")

        elif isinstance(level1_value, list):
            # 处理值是列表的直接属性（如colors）
            # 决定用户是否偏好这个一级属性下的某些值
            # 这里随机选择列表中的一部分值
            select_attribute = random.sample(level1_value, 1)[0]
            # 同样可以添加偏好强度
            user_preferences.append(f"{level1_key} is {select_attribute}")
            # 处理值是字符串、数字等其他类型的直接属性（如果存在）
            # 这里简单将其作为偏好，并添加强度
    # print(", ".join(user_preferences))
    return ", ".join(user_preferences)
# def disp_loss(z):  # Dispersive Loss implementation (InfoNCE-L2 variant)
#     z = z.reshape((z.shape[0], -1))  # flatten
#     diff = torch.nn.functional.pdist(z).pow(2) / z.shape[1]  # pairwise distance
#     diff = torch.concat((diff, diff, torch.zeros(z.shape[0]).cuda()))  # match JAX implementation of full BxB matrix
#     return torch.log(torch.exp(-diff).mean())  # calculate loss



# === 关键：在任何CUDA导入前执行 ===
def fix_cuda_device_count():
    """绕过驱动层设备计数错误"""
    try:

        actual_count = 7

        # 2. 创建虚拟设备计数函数
        class CUDADeviceCount:
            def __init__(self, count):
                self.count = count
                self.original = None

            def __enter__(self):
                try:
                    # 加载CUDA运行时库
                    libcudart = ctypes.CDLL("libcudart.so")

                    # 保存原始函数
                    self.original = libcudart.cudaGetDeviceCount

                    # 设置函数原型
                    libcudart.cudaGetDeviceCount.restype = ctypes.c_int
                    libcudart.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]

                    # 创建mock函数
                    def mock_cudaGetDeviceCount(count_ptr):
                        count_ptr.contents.value = self.count
                        return 0  # CUDA_SUCCESS

                    # 转换为C函数
                    CMPFUNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int))
                    self.mock_func = CMPFUNC(mock_cudaGetDeviceCount)

                    # 替换函数
                    libcudart.cudaGetDeviceCount = self.mock_func
                    return self

                except Exception as e:
                    print(f"⚠️ CUDA函数替换失败: {e}")
                    return None

            def __exit__(self, exc_type, exc_val, exc_tb):
                # 恢复原始函数（安全措施）
                if self.original:
                    try:
                        libcudart = ctypes.CDLL("libcudart.so")
                        libcudart.cudaGetDeviceCount = self.original
                    except:
                        pass

        # 3. 应用虚拟设备计数
        return CUDADeviceCount(actual_count)

    except Exception as e:
        print(f"⚠️ 设备修复失败: {e}")
        return None


