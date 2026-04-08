import ctypes
import json
import os
import sys

sys.path.append("/hdd5/wangzihao/Lab/OminiControl")

import pandas as pd
import torch
from diffusers import FluxPipeline  # 注意：可能需要根据实际安装的库调整


def main():
    # 1. 加载基础模型
    pipeline = FluxPipeline.from_pretrained(
        "../../.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to("cuda")
    # target_modules = [
    #     "attn.to_k",
    #     "attn.to_q",
    #     "attn.to_v",
    #     "attn.to_out.0",
    #     "attn.add_k_proj",
    #     "attn.add_q_proj",
    #     "attn.add_v_proj",
    #     "attn.to_add_out",
    #     "ff.net.0.proj",
    #     "ff.net.2",
    #     "ff_context.net.0.proj",
    #     "ff_context.net.2",
    # ]
    # transformer_lora_config = LoraConfig(
    #     r=1,
    #     lora_alpha=1,
    #     init_lora_weights="gaussian",
    #     target_modules=target_modules,
    # )
    # pipeline.transformer.add_adapter(transformer_lora_config)
    # 2. 加载LoRA权重 (本地文件方式)

    # with open("../../data/user_data/user.json") as f:
    #     user_id_list = json.load(f)
    user_index_list = ["test"]
    prompts_path_format = "../../data/test/test_test.csv"
    # 3. 设置LoRA缩放权重 (根据搜索结果[[7]]中提到的weight参数)
    lora_scale = 1.0  # 可根据效果调整，通常在0.5-1.0之间
    save_dir_format = "evaluation_outputs/LoRA/{}_test"
    for user_id in user_index_list:
        pipeline.load_lora_weights(
            f"/hdd5/wangzihao/Lab/DiffusionDPO/models/user-test1-flux-lora",  # LoRA文件所在目录
            weight_name="pytorch_lora_weights.safetensors",  # 您的LoRA文件名
        )
        df = pd.read_csv(prompts_path_format.format(user_id), index_col=0)
        print(user_id)
        image_dir = save_dir_format.format(user_id)
        os.makedirs(image_dir, exist_ok=True)
        for index, row in df.iterrows():
            prompt = str(row.caption)

            case_number = int(str(index))
            for seed in range(30):
                # 4. 生成图像
                image = pipeline(
                    prompt=prompt,
                    num_inference_steps=28,
                    height=1024,
                    width=1024,
                    guidance_scale=3,
                ).images[0]
                os.makedirs(f"evaluation_outputs/LoRA/{user_id}_test/", exist_ok=True)
                # 5. 保存图像
                image.save(f"evaluation_outputs/LoRA/{user_id}_test/{case_number}_{seed}.png")


def generate_ori():
    # 1. 加载基础模型
    pipeline = FluxPipeline.from_pretrained(
        "../../.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to("cuda")
    # 2. 加载LoRA权重 (本地文件方式)


    csv_names = ["test"]

    # 3. 设置LoRA缩放权重 (根据搜索结果[[7]]中提到的weight参数)

    save_dir_format = "evaluation_outputs/flux/{}_test"
    csv_path_format = "../../data/{}/{}.csv"
    for user_id in csv_names:
        # pipeline.load_lora_weights(
        #     f"runs/LoRA/user-{user_id}-flux-lora/",  # LoRA文件所在目录
        #     weight_name="pytorch_lora_weights.safetensors"  # 您的LoRA文件名
        # )
        if user_id=="mrfz":
            df = pd.read_csv(csv_path_format.format(user_id, user_id).replace(".csv", "_test.csv"), index_col=0)
        else:
            df = pd.read_csv(csv_path_format.format(user_id, user_id), index_col=0)
        print(user_id)
        image_dir = save_dir_format.format(user_id)
        os.makedirs(image_dir, exist_ok=True)
        for index, row in df.iterrows():
            prompt = str(row.caption)

            case_number = int(str(index))
            for seed in range(1,20):
                # 4. 生成图像
                generator = torch.Generator("cuda").manual_seed(seed)
                image = pipeline(
                    prompt=prompt,
                    num_inference_steps=28,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    generator=generator
                ).images[0]

                # 5. 保存图像
                image.save(os.path.join(image_dir, f"{case_number}_{seed}.png"))

if __name__ == "__main__":
    main()
