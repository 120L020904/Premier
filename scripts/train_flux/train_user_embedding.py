import gc
import json
import os
import random
import re
import sys
import time

import pandas as pd
from torch import nn

from scripts.dataset.PIP_dataset import ImagePromptDataset

sys.path.append("/home/disk1/wangzihao/nfs-27/dxm/wangzihao_dxm/OminiControl")
from scripts.pipeline.flux_adapter import transformer_forward_verse
from scripts.train_flux.adapter_trainer import get_config, train

from torch.utils.data import Dataset
import torchvision.transforms as T

from scripts.utils.utils import get_image_dxm_csv
import os.path

import prodigyopt
import torch
from diffusers import FluxPipeline
from safetensors.torch import save_file
from scripts.pipeline.flux_omini import encode_images
from scripts.pipeline.mod_adapters import load_modulation_adapter
import lightning as L


@torch.no_grad()
def empty_test_function(model, save_path, file_name):
    return


class CustomDataset(Dataset):
    def __init__(
            self,
            condition_size=(512, 512),
            target_size=(512, 512),
            drop_text_prob: float = 0,
            drop_image_prob: float = 0,
            return_pil_image: bool = False,
            position_scale=1.0,
            train_df_path_format="../pickapic_dxm_coco/split/{}.csv",
            user_idx=-1,
            image_dir=""
    ):
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale
        self.image_dir = image_dir
        train_df_path = train_df_path_format.format(user_idx)
        print(train_df_path)
        self.train_df = pd.read_csv(train_df_path)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx, ):
        image_data = self.train_df.iloc[idx]

        images = get_image_dxm_csv([image_data], image_dir=self.image_dir)

        image = images[0]

        image = image.resize(self.target_size).convert("RGB")
        if type(image_data["caption"]) is not str:
            caption = ""
        else:
            caption = image_data["caption"]

        description = re.sub(r'[^a-zA-Z0-9\s]', '', caption)

        if random.random() < self.drop_text_prob:
            description = ""
        return_dict = {"image": self.to_tensor(image), "description": description, "idx": idx}

        return return_dict


class OminiModelUserEmbedding(L.LightningModule):
    def __init__(
            self,
            flux_pipe_id: str,
            model_path: str = None,
            device: str = "cuda",
            dtype: torch.dtype = torch.bfloat16,
            model_config: dict = {},
            optimizer_config: dict = None,
            user_idx=-1,
            gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
            flux_pipe_id, torch_dtype=dtype, local_files_only=True
        ).to(device)
        self.init_pipe()
        self.user_idx = user_idx
        # Initialize LoRA layers
        # self.model_config["model"]["modulation"]["uncond"] = True
        self.train_layers = self.init_mod_adapter(model_path=model_path,
                                                  config=model_config,
                                                  token_num=model_config["model"]["modulation"]["user_token_num"],
                                                  device=device,
                                                  dtype=dtype)

        self.is_uncond = model_config["model"]["modulation"].get("uncond", False)

        (
            self.uncond_embeds,
            _,
            _,
        ) = self.flux_pipe.encode_prompt(
            prompt="",
            prompt_2="",
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=self.flux_pipe.device,
            num_images_per_prompt=1,
            max_sequence_length=self.model_config.get("max_sequence_length", 512),
            lora_scale=None,
        )
        self.to(device).to(dtype)

    def init_pipe(self):
        # Freeze the Flux pipeline
        self.flux_pipe.transformer.requires_grad_(False).eval()
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

    def init_mod_adapter(self, model_path: str = None,
                         config=None,
                         device: str = "cuda",
                         dtype: torch.dtype = torch.bfloat16,
                         user_num: int = 1,
                         token_num: int = 30):
        self.mod_adapter = load_modulation_adapter(config, dtype, device, ckpt_dir=model_path, is_training=True)
        self.mod_adapter.requires_grad_(False).eval()
        self.token_num = token_num
        self.user_embedding = nn.Embedding(num_embeddings=user_num,
                                           embedding_dim=token_num * 1024,
                                           ).to(device=device, dtype=dtype)
        self.user_embedding.weight.data.uniform_(-0.001, 0.001)
        # self.user_embedding_param.grad = torch.zeros_like(self.user_embedding_param.data)  # 初始化为零
        self.user_embedding.train()
        self.user_embedding.requires_grad_(True)
        if self.optimizer_config["type"] == "AdamW":
            param_list = [{"params": list(self.user_embedding.parameters()), "lr": 0.01}]
        else:
            param_list = [{"params": list(self.user_embedding.parameters())}]
        return param_list

    def save_lora(self, path: str):
        os.makedirs(path, exist_ok=True)

        user_state_dict = self.user_embedding.state_dict()
        save_file(
            tensors=user_state_dict,
            filename=os.path.join(path, f"user_embedding_{self.user_idx}.safetensors"),
        )

    def configure_optimizers(self):
        # Freeze the transformer
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.train_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            for param in p["params"]:
                param.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(
                self.trainable_params,
                **opt_config["params"]
            )
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError("Optimizer not implemented.")
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs, ids, prompts = batch["image"], batch["idx"], batch["description"]
        image_latent_mask = batch.get("image_latent_mask", None)
        ids = torch.tensor([0] * len(batch["idx"])).to(self.device)
        user_preference_embedding = self.user_embedding(ids).view(-1, self.token_num, 1024)
        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)

            # Prepare text input
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.flux_pipe.encode_prompt(
                prompt=prompts,
                prompt_2=prompts,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.flux_pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("max_sequence_length", 512),
                lora_scale=None,
            )

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device) + 1.0)
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
            if image_latent_mask is not None:
                x_0 = x_0[:, image_latent_mask[0]]
                x_1 = x_1[:, image_latent_mask[0]]
                x_t = x_t[:, image_latent_mask[0]]
                img_ids = img_ids[image_latent_mask[0]]

            # Prepare conditions
            condition_latents, condition_ids = [], []

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.flux_pipe.transformer.config.guidance_embeds
                else None
            )
        if self.is_uncond:
            uncond_embeds = self.uncond_embeds.repeat(len(prompts), 1, 1).detach()
            delta_emb, delta_emb_pblock = self.mod_adapter(t, prompt_embeds, user_preference_embedding, uncond_embeds)
        else:
            delta_emb, delta_emb_pblock = self.mod_adapter(t, prompt_embeds, user_preference_embedding)
        # Forward pass
        transformer_out = transformer_forward_verse(
            self.flux_pipe.transformer,
            model_config=self.model_config,
            hidden_states=x_t,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            img_ids=img_ids,
            txt_ids=text_ids,
            guidance=guidance,
            timestep=t,
            return_dict=False,
            delta_emb=delta_emb,
            delta_emb_pblock=delta_emb_pblock,
            # There are three timesteps for the three branches
            # (text, image, and the condition)
        )

        pred = transformer_out[0]
        # Compute loss
        step_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()

        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def generate_a_sample(self):
        raise NotImplementedError("Generate a sample not implemented.")


def cuda_clear():
    # 强制垃圾回收
    gc.collect()

    # 清空CUDA缓存
    torch.cuda.empty_cache()


def main():
    # Initialize

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    # Initialize model
    with open("../../data/user_data/user_long.json") as f:
        user_id_list = json.load(f)
    user_index_list = list(range(25))
    
    model_name = "20250829-215051"
    experient_name = "xverse_test2_cross_dips_ori"
    step = 260000
    save_path = f"../Premier/runs/{model_name}"
    model_save_path = f"{save_path}/ckpt/{step}"
    config_path = f"train/config/premier_user.yaml"
    config = get_config(config_path)
    config["adapter_path"] = save_path
    training_config = config["train"]
    adapter_config_path = f"{save_path}/adapter_config.yaml"
    adapter_config = get_config(config_path=adapter_config_path)
    adapter_config["model"]["modulation"]["uncond"] = False
    drop_text_prob = training_config["dataset"]["drop_text_prob"]
    run_name = time.strftime("%Y%m%d-%H%M%S")
    # run_name = "20251014-143621"
    for user_idx in user_index_list:
        # Initialize custom dataset
        print(f"user_idx {user_idx}")
        print(f"model path {model_save_path}")
        dataset = CustomDataset(user_idx=user_idx, drop_text_prob=drop_text_prob,
                                train_df_path_format=config.get("csv_path",
                                                                "../../data/pickapic_dxm_coco/split/{}.csv"),
                                image_dir=config.get("data_path").format(user_idx))
        trainable_model = OminiModelUserEmbedding(
            model_path=model_save_path,
            flux_pipe_id=config["flux_path"],
            device=f"cuda",
            dtype=getattr(torch, config["dtype"]),
            optimizer_config=training_config["optimizer"],
            model_config=adapter_config,
            gradient_checkpointing=training_config.get("gradient_checkpointing", False),
            user_idx=user_idx
        )

        train(dataset, trainable_model, config=config, adapter_config=adapter_config, test_function=empty_test_function,
              run_name=run_name)
        del trainable_model
        cuda_clear()


def train_PIP():
    # Initialize

    torch.cuda.set_device("cuda:0")
    # Initialize model
    data_dir = "/hdd5/wangzihao/data/PIP_user_split_dataset"
    image_dir = "/hdd5/wangzihao/data/PIP-dataset-image"
    jsonl_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_train.jsonl')])[25:]
    model_name = "20250829-215051"
    step = 260000
    save_path = f"../OminiControl/runs/{model_name}"
    model_save_path = f"{save_path}/ckpt/{step}"
    config_path = f"train/config/personalize_xverse_user.yaml"

    config = get_config(config_path)
    config["adapter_path"] = save_path
    training_config = config["train"]
    adapter_config_path = f"{save_path}/adapter_config.yaml"
    adapter_config = get_config(config_path=adapter_config_path)
    adapter_config["model"]["modulation"]["uncond"] = False
    drop_text_prob = training_config["dataset"]["drop_text_prob"]
    run_name = time.strftime("%Y%m%d-%H%M%S")
    # run_name = "20251014-143621"
    for jsonl_file in jsonl_files:
        # Initialize custom dataset
        print(f"user_idx {jsonl_file}")
        print(f"model path {model_save_path}")
        user_id = jsonl_file.split("_")[0]
        jsonl_path = os.path.join(data_dir, jsonl_file)
        dataset = ImagePromptDataset(jsonl_path=jsonl_path, image_dir=image_dir, drop_text_prob=drop_text_prob)
        trainable_model = OminiModelUserEmbedding(
            model_path=model_save_path,
            flux_pipe_id=config["flux_path"],
            device=f"cuda",
            dtype=getattr(torch, config["dtype"]),
            optimizer_config=training_config["optimizer"],
            model_config=adapter_config,
            gradient_checkpointing=training_config.get("gradient_checkpointing", False),
            user_idx=user_id
        )

        train(dataset, trainable_model, config=config, adapter_config=adapter_config, test_function=empty_test_function,
              run_name=run_name)
        del trainable_model
        cuda_clear()


if __name__ == "__main__":
    train_PIP()
