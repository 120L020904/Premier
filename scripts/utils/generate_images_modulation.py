
import json
import os
import shutil
import sys
import pandas as pd


import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file
from torch import nn



sys.path.append("/hdd5/wangzihao/Lab/OminiControl")
sys.path.append("/home/disk4/wangzihao/dxm/wangzihao_dxm/OminiControl")
from scripts.train_flux.train_user_embedding_linear import EmbeddingLinearCombination

from scripts.pipeline.flux_adapter import generate_xverse
from scripts.pipeline.mod_adapters import load_modulation_adapter
from scripts.utils.utils import save_images, get_config

local_files_only = True

devices = ["cuda:0", "cuda:1"]


@torch.no_grad()
def generate_images_modulation(flux_pipe,
                               mod_adapter,
                               user_preference_embedding: torch.Tensor,
                               model_config,
                               prompts_path,
                               model_name,
                               run_name,
                               save_path,
                               guidance_scale=3.5,
                               image_size=512,
                               inference_steps=30,
                               is_ori=True
                               ):
    df = pd.read_csv(prompts_path, index_col=0)

    folder_path = f'{save_path}/image_guide_{guidance_scale}_size_{image_size}_{model_name}_{run_name}'
    os.makedirs(folder_path, exist_ok=True)
    print(folder_path)

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            print(f"正在删除目录: {item_path}")
            shutil.rmtree(item_path)  # 递归删除整个目录树

    for ori_seed in range(0, 20):
        seed = ori_seed
        for index, row in df.iterrows():
            prompt = str(row.caption)
            if "seed" in row:
                seed = int(row["seed"])

                # print(row["seed"])
            else:
                seed += 1
            generator = torch.Generator(devices[0]).manual_seed(seed)
            image_dir = folder_path
            os.makedirs(image_dir, exist_ok=True)
            case_number = int(str(index))
            # if os.path.isfile(image_path) and not is_image_corrupted(image_path):
            #     continue
            num_inference_steps = inference_steps  # Number of denoising steps
            print(prompts_path)

            # print(f"score {score.item()}")

            pil_images = generate_xverse(pipeline=flux_pipe,
                                         mod_adapter=mod_adapter,
                                         user_preference_embedding=user_preference_embedding,
                                         prompt_2=prompt,
                                         prompt=prompt,
                                         num_inference_steps=num_inference_steps,
                                         num_images_per_prompt=1,
                                         guidance_scale=guidance_scale,
                                         generator=generator,
                                         height=image_size,
                                         width=image_size,
                                         model_config=model_config,
                                         is_ori=is_ori
                                         ).images

            save_images(pil_images=pil_images,
                        folder_path=image_dir,
                        case_number=case_number,
                        seed=ori_seed)

            del pil_images
    return


def main():
    # Initialize /home/disk3/wangzihao/OminiControl/runs/xverse_test_dips_stable/20251017-172159/ckpt/1000000
    dtype = torch.bfloat16
    model_name = "20251017-172159"
    experient_name = "xverse_test_dips_stable"
    step = 1000000
    save_path = f"/home/disk3/wangzihao/OminiControl/runs/{experient_name}/{model_name}"
    model_save_path = f"{save_path}/ckpt/{step}"
    # config_path = "/home/disk3/wangzihao/OminiControl/runs/xverse_test_dips_ori/20251002-174651/ckpt/700000"
    adapter_config_path = f"{save_path}/adapter_config.yaml"
    config = get_config()

    adapter_config = get_config(config_path=adapter_config_path)
    model_config = adapter_config
    with open("../pickapic_dxm_coco/json/user.json") as f:
        user_id_list = json.load(f)
    user_index_list = user_id_list[:16]
    save_dir = "/home/disk1/wangzihao/OminiControl/evaluation_outputs/{}_test"
    # need to change
    prompts_path = "../pickapic_dxm_coco/split/{}_test.csv"
    pipe = FluxPipeline.from_pretrained(
        config["flux_path"], torch_dtype=dtype
    ).to(devices[0])
    pipe.unload_lora_weights()
    
    mod_adapter = load_modulation_adapter(adapter_config, dtype, devices[0], ckpt_dir=model_save_path,
                                            is_training=True)
    mod_adapter.eval()
    print(type(mod_adapter))
    state_dict = load_file(os.path.join(model_save_path, "user_embedding.safetensors"))
    user_token_num = adapter_config["model"]["modulation"]["user_token_num"]
    user_embedding = nn.Embedding(num_embeddings=1001,
                                  embedding_dim=user_token_num * 1024,
                                  ).to(device=devices[0], dtype=dtype)
    # user_embedding = nn.Embedding(num_embeddings=150,
    #                               embedding_dim=10 * 1024,
    #                               ).to(device=devices[0], dtype=dtype)
    user_embedding.load_state_dict(state_dict)
    # base_name = os.path.basename(model_save_dir)
    model_name = model_name + f"-{step}"
    
    for user_id in user_index_list:
        indices = torch.tensor([user_id], dtype=torch.long).to(device=devices[0])
        print(indices)
        generate_images_modulation(
            pipe,
            mod_adapter=mod_adapter,
            user_preference_embedding=user_embedding(indices).view(-1, user_token_num, 1024),
            model_config=model_config,
            prompts_path=prompts_path.format(user_id),
            model_name=model_name,
            run_name="",
            guidance_scale=2.5,
            inference_steps=30,
            image_size=512,
            save_path=save_dir.format(user_id),
        )


def test_user():
    # Initialize

    dtype = torch.bfloat16
    model_name = "20250829-215051"
    step = 260000
    # model_save_path = f"xverse_b8_user_prodigy_uncond_32/20251013-185804/ckpt/5000"
    run_name = "20251028-191801"
    user_step = 5000
    user_run_name = f"{run_name}-{user_step}"
    user_save_dir = f"runs/xverse_b8_user_prodigy_paokemen/{run_name}"
    user_save_path = f"{user_save_dir}/ckpt/{user_step}"

    # config_path = f"/home/nfs/nfs-160/wangzihao/OminiControl/runs/xverse_b8_user_prodigy_uncond_sparse/20250918-122550/ckpt/5000"

    save_path = f"runs/{model_name}/"
    adapter_model_save_path = f"{save_path}/ckpt/{step}"
    config_path = f"{user_save_dir}/config.yaml"
    adapter_config_path = f"{save_path}/adapter_config.yaml"
    config = get_config()
    adapter_config = get_config(config_path=adapter_config_path)
    # adapter_config["model"]["modulation"]["uncond"] = False
    model_config = adapter_config
    with open("../pickapic_dxm_coco/json/user.json") as f:
        user_id_list = json.load(f)
    user_index_list = user_id_list[3000:3050]
    # user_index_list = list(range(16))
    save_dir = "evaluation_outputs/{}_test/"
    # need to change
    prompts_path = "../../data/mrfz/mrfz_test.csv"

    pipe = FluxPipeline.from_pretrained(
        config["flux_path"], torch_dtype=dtype,
    ).to(devices[0])
    pipe.unload_lora_weights()
    mod_adapter = load_modulation_adapter(model_config, dtype, devices[0], ckpt_dir=adapter_model_save_path,
                                          is_training=True)
    print(type(mod_adapter))
    mod_adapter.eval()
    user_token_num = 30
    is_linear = False
    if is_linear:
        train_user_embedding = nn.Embedding(num_embeddings=1000,
                                            embedding_dim=user_token_num * 1024,
                                            ).to(device=devices[0], dtype=dtype)
    else:
        train_user_embedding = nn.Embedding(num_embeddings=1,
                                            embedding_dim=user_token_num * 1024,
                                            ).to(device=devices[0], dtype=dtype)
    # user_embedding = nn.Embedding(num_embeddings=150,
    #                               embedding_dim=10 * 1024,
    #                               ).to(device=devices[0], dtype=dtype)

    model_name = model_name + f"-{step}"

    for user_id in user_index_list:
        indices = torch.tensor([0], dtype=torch.long).to(device=devices[0])

        if is_linear:
            user_path = os.path.join(user_save_path,
                                     f"user_combination_{user_id}.safetensors")
            train_user_path = os.path.join(adapter_model_save_path, f"user_embedding.safetensors")
            state_dict = load_file(train_user_path)
            train_user_embedding.load_state_dict(state_dict)
            combination_state_dict = load_file(user_path)
            embeddingCombination = EmbeddingLinearCombination(combination_size=1, embedding_num=1000,
                                                              use_softmax=False).to(device=devices[0], dtype=dtype)
            embeddingCombination.load_state_dict(combination_state_dict)
            weights = embeddingCombination.get_combination_weights()
            user_embedding = embeddingCombination(train_user_embedding, input_ids=indices).view(-1, user_token_num,
                                                                                                1024)
        else:
            train_user_path = os.path.join(user_save_path, f"user_embedding_{user_id}.safetensors")
            state_dict = load_file(train_user_path)
            train_user_embedding.load_state_dict(state_dict)
            user_embedding = train_user_embedding(indices).view(-1, user_token_num, 1024)
        generate_images_modulation(
            pipe,
            mod_adapter=mod_adapter,
            user_preference_embedding=user_embedding,
            model_config=model_config,
            prompts_path=prompts_path.format(user_id),
            model_name=model_name,
            run_name=user_run_name,
            guidance_scale=2.5,
            inference_steps=30,
            image_size=512,
            save_path=save_dir.format(user_id),
        )


def test_user_same_csv():
    # Initialize

    dtype = torch.bfloat16
    model_name = "20250829-215051"
    step = 260000
    # model_save_path = f"xverse_b8_user_prodigy_uncond_32/20251013-185804/ckpt/5000"
    run_name = "20251119-145711"
    user_step = 5000
    user_run_name = f"{run_name}-{user_step}"
    user_save_dir = f"runs/xverse_b8_user_prodigy_wzry/{run_name}/"
    user_save_path = f"{user_save_dir}/ckpt/{user_step}"
    # config_path = f"/home/nfs/nfs-160/wangzihao/OminiControl/runs/xverse_b8_user_prodigy_uncond_sparse/20250918-122550/ckpt/5000"
    save_path = f"runs/{model_name}/"
    adapter_model_save_path = f"{save_path}/ckpt/{step}"
    config_path = f"{user_save_dir}/config.yaml"
    adapter_config_path = f"{save_path}/adapter_config.yaml"
    config = get_config()
    adapter_config = get_config(config_path=adapter_config_path)
    # adapter_config["model"]["modulation"]["uncond"] = False
    model_config = adapter_config
    with open("../../data/user_data/user.json") as f:
        user_id_list = json.load(f)
    user_index_list = ["wzry"]
    # user_index_list = list(range(16))
    save_dir = "evaluation_outputs/{}_test/"
    # need to change 17985 18037
    prompts_path = "../../data/wzry/wzry.csv"
    pipe = FluxPipeline.from_pretrained(
        config["flux_path"], torch_dtype=dtype,
    ).to(devices[0])
    pipe.unload_lora_weights()
    mod_adapter = load_modulation_adapter(model_config, dtype, devices[0], ckpt_dir=adapter_model_save_path,
                                          is_training=True)
    print(type(mod_adapter))
    mod_adapter.eval()
    user_token_num = 30
    is_linear = False
    if is_linear:
        train_user_embedding = nn.Embedding(num_embeddings=1000,
                                            embedding_dim=user_token_num * 1024,
                                            ).to(device=devices[0], dtype=dtype)
    else:
        train_user_embedding = nn.Embedding(num_embeddings=1,
                                            embedding_dim=user_token_num * 1024,
                                            ).to(device=devices[0], dtype=dtype)
    # user_embedding = nn.Embedding(num_embeddings=150,
    #                               embedding_dim=10 * 1024,
    #                               ).to(device=devices[0], dtype=dtype)

    model_name = model_name + f"-{step}"

    for user_id in user_index_list:
        indices = torch.tensor([0], dtype=torch.long).to(device=devices[0])

        if is_linear:
            user_path = os.path.join(user_save_path,
                                     f"user_combination_{user_id}.safetensors")
            train_user_path = os.path.join(adapter_model_save_path, f"user_embedding.safetensors")
            state_dict = load_file(train_user_path)
            train_user_embedding.load_state_dict(state_dict)
            combination_state_dict = load_file(user_path)
            embeddingCombination = EmbeddingLinearCombination(combination_size=1, embedding_num=1000,
                                                              use_softmax=False).to(device=devices[0], dtype=dtype)
            embeddingCombination.load_state_dict(combination_state_dict)
            weights = embeddingCombination.get_combination_weights()
            user_embedding = embeddingCombination(train_user_embedding, input_ids=indices).view(-1, user_token_num,
                                                                                                1024)
        else:
            train_user_path = os.path.join(user_save_path, f"user_embedding_{user_id}.safetensors")
            state_dict = load_file(train_user_path)
            train_user_embedding.load_state_dict(state_dict)
            user_embedding = train_user_embedding(indices).view(-1, user_token_num, 1024)
        # macs, params = get_model_complexity_info(mod_adapter, tuple(user_embedding.size()),
        #                                          as_strings=True,
        #                                          print_per_layer_stat=True)
        # print(f"macs {macs}, params {params}")
        generate_images_modulation(
            pipe,
            mod_adapter=mod_adapter,
            user_preference_embedding=user_embedding,
            model_config=model_config,
            prompts_path=prompts_path.format(user_id),
            model_name=model_name,
            run_name=user_run_name,
            guidance_scale=2.5,
            inference_steps=30,
            image_size=512,
            save_path=save_dir.format(user_id),
        )


if __name__ == "__main__":
    try:
        test_user_same_csv()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(e)
            sys.exit(110)
        else:
            raise e
