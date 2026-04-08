import itertools
import json
import os
import pickle
import random
import sys

from torch.cuda.amp import autocast

sys.path.append("/home/nfs/nfs-27/wangzihao/dxm/wangzihao_dxm/OminiControl")
from scripts.pipeline.flux_adapter import transformer_forward_verse
from scripts.pipeline.mod_adapters import load_modulation_adapter
from scripts.train_flux.adapter_trainer import train, get_config
from torch.utils.data import Dataset
import torchvision.transforms as T

from scripts.utils.utils import get_image_dxm, get_image_dxm_release, stable_disp_loss, disp_loss, AllGather, distributed_disp_loss
import os.path

import prodigyopt
import torch
from diffusers import FluxPipeline
from safetensors.torch import save_file, load_file

from scripts.pipeline.flux_omini import encode_images
import lightning as L
import torch.nn as nn


@torch.no_grad()
def empty_test_function(model, save_path, file_name):
    return


class CustomDataset(Dataset):
    def __init__(
            self,
            pkl_path,
            image_dir,
            json_path,
            condition_size=(512, 512),
            target_size=(512, 512),
            drop_text_prob: float = 0,
            drop_image_prob: float = 0,
            return_pil_image: bool = False,
            position_scale=1.0,
            delta_save_path_format="",
            user_num=10000
    ):
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_dir = image_dir
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale
        with open(pkl_path, 'rb') as f:
            data_ = pickle.load(f)
        with open(json_path) as f:
            user_id_list = json.load(f)
        self.user_index_list = user_id_list[:user_num]
        self.data = list(itertools.islice(data_.values(), int(self.user_index_list[-1]) + 1))

        self.to_tensor = T.ToTensor()
        self.delta_save_path_format = delta_save_path_format

    def __len__(self):
        return len(self.user_index_list)

    def __getitem__(self, idx):
        user_idx = self.user_index_list[idx]
        image_data = self.data[user_idx][:-1]

        image_item_list = random.sample(image_data, 1)
        # images = get_image_dxm(image_item_list, image_dir="/home/nfs/nfs-55/mowenyi/data/flux_coco_49")
        images = get_image_dxm_release(image_item_list, image_dir=self.image_dir)
        image_item = image_item_list[0]
        image = images[0]
        image = image.resize(self.target_size).convert("RGB")
        description = image_item[2]
        drop_text = random.random() < self.drop_text_prob
        if drop_text:
            description = ""
        return_dict = {"image": self.to_tensor(image), "description": description, "idx": idx}

        return return_dict


class OminiModelDeltaVerse(L.LightningModule):
    def __init__(
            self,
            flux_pipe_id: str,
            model_path: str = None,
            device: str = "cuda",
            dtype: torch.dtype = torch.bfloat16,
            model_config: dict = {},
            optimizer_config: dict = None,
            user_num: int = 10000,
            dips_strength=0.1,
            is_dips=True,
            dips_stable=False,
            dips_uncond=True,
            drop_user=0.1,
            drop_text=0.1,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.drop_user_prob = drop_user
        self.drop_text_prob = drop_text
        self.user_num = user_num
        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
            flux_pipe_id, torch_dtype=dtype
        ).to(device)
        self.flux_pipe.transformer.train()
        self.mod_adapter = load_modulation_adapter(model_config, dtype, device, ckpt_dir=model_path, is_training=True)
        self.token_num = model_config["model"]["modulation"]["user_token_num"]
        self.user_embedding = nn.Embedding(num_embeddings=self.user_num + 1,
                                           embedding_dim=self.token_num * 1024,
                                           ).to(device=device, dtype=dtype)
        self.is_uncond = model_config["model"]["modulation"].get("uncond", False)
        self.dips_uncond = model_config["model"]["modulation"].get("dips_uncond", False)
        self.no_text = model_config["model"]["modulation"].get("no_text", False)

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
        self.user_embedding.weight.data.uniform_(-0.1, 0.1)
        self.init_pipe()
        # Initialize LoRA layers
        self.train_layers = self.init_mod_adapter(model_path=model_path, )
        self.is_dips = is_dips
        self.dips_stable = dips_stable
        print(f"self.is_dips {self.is_dips}, self.is_uncond {self.is_uncond}, self.dips_stable {self.dips_stable}")
        self.dips_strength = dips_strength
        self.all_gather = AllGather.apply
        self.to(device).to(dtype)

    def init_pipe(self):
        # Freeze the Flux pipeline
        self.flux_pipe.transformer.requires_grad_(False).eval()
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

    def init_mod_adapter(self, model_path: str = None):

        self.mod_adapter.train()
        self.mod_adapter.requires_grad_(True)

        if model_path is not None:
            print(f"load user weight from {model_path}")
            state_dict = load_file(os.path.join(model_path, "user_embedding.safetensors"))
            self.user_embedding.load_state_dict(state_dict)
        # self.user_embedding_param.grad = torch.zeros_like(self.user_embedding_param.data)  # 初始化为零
        self.user_embedding.train()
        self.user_embedding.requires_grad_(True)
        if self.optimizer_config["type"] == "AdamW":

            param_list = [{"params": list(self.mod_adapter.parameters()), "lr": 0.0005},
                          {"params": list(self.user_embedding.parameters()), "lr": 0.1}, ]
        else:
            param_list = [{"params": list(self.mod_adapter.parameters()), },
                          {"params": list(self.user_embedding.parameters()), }, ]
        return param_list

    def save_lora(self, path: str):
        print(path)
        os.makedirs(path, exist_ok=True)
        state_dict = self.mod_adapter.state_dict()

        # 保存到 Safetensors 文件（可添加元数据）
        save_file(
            tensors=state_dict,
            filename=os.path.join(path, "mod_adapter.safetensors"),
        )
        user_state_dict = self.user_embedding.state_dict()
        save_file(
            tensors=user_state_dict,
            filename=os.path.join(path, "user_embedding.safetensors"),
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
                params=self.trainable_params,
                **opt_config["params"]
            )
            return optimizer
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                params=self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(params=self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError("Optimizer not implemented.")

        return optimizer

    def training_step(self, batch, batch_idx):
        imgs, ids, prompts = batch["image"], batch["idx"], batch["description"]
        image_latent_mask = batch.get("image_latent_mask", None)
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

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.flux_pipe.transformer.config.guidance_embeds
                else None
            )
        assert not torch.isnan(user_preference_embedding).any(), "Input user data contains NaN!"
        assert not torch.isnan(prompt_embeds).any(), "Input prompt contains NaN!"
        # torch.autograd.set_detect_anomaly(True)
        if self.is_uncond:
            uncond_embeds = self.uncond_embeds.repeat(len(prompts), 1, 1).detach()
            delta_emb, delta_emb_pblock = self.mod_adapter(t, prompt_embeds,
                                                           user_preference_embedding,
                                                           uncond_embeds)
        elif self.no_text:
            uncond_embeds = self.uncond_embeds.repeat(len(prompts), 1, 1).detach()
            delta_emb, delta_emb_pblock = self.mod_adapter(t,  uncond_embeds,
                                                           user_preference_embedding)
        elif self.dips_uncond:
            uncond_embeds = self.uncond_embeds.repeat(len(prompts), 1, 1).detach()
            delta_emb, delta_emb_pblock = self.mod_adapter(t, prompt_embeds,
                                                           user_preference_embedding)
            delta_emb_uncond, delta_emb_pblock_uncond = self.mod_adapter(t, uncond_embeds,
                                                           user_preference_embedding)
        else:
            delta_emb, delta_emb_pblock = self.mod_adapter(t, prompt_embeds,
                                                           user_preference_embedding)
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
        diff_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.diff_loss = diff_loss.item()
        self.last_t = t.mean().item()
        self.ids = ids
        if self.is_dips:
            dips_loss_delta = torch.tensor([0]).to(dtype=self.dtype, device=self.device)
            dips_loss_delta_p = torch.tensor([0]).to(dtype=self.dtype, device=self.device)
            if self.dips_stable:
                if self.dips_uncond:
                    if delta_emb is not None:
                        dips_loss_delta = distributed_disp_loss(self.all_gather, delta_emb_uncond)
                    if delta_emb_pblock is not None:
                        dips_loss_delta_p = distributed_disp_loss(self.all_gather, delta_emb_pblock_uncond)
                else:
                    if delta_emb is not None:
                        dips_loss_delta = distributed_disp_loss(self.all_gather, delta_emb)
                    if delta_emb_pblock is not None:
                        dips_loss_delta_p = distributed_disp_loss(self.all_gather, delta_emb_pblock)
            else:
                if self.dips_uncond:
                    if delta_emb is not None:
                        dips_loss_delta = disp_loss(delta_emb_uncond)
                    if delta_emb_pblock is not None:
                        dips_loss_delta_p = disp_loss(delta_emb_pblock_uncond)
                else:
                    if delta_emb is not None:
                        dips_loss_delta = disp_loss(delta_emb)
                    if delta_emb_pblock is not None:
                        dips_loss_delta_p = disp_loss(delta_emb_pblock)
            self.dips_loss = (self.dips_strength * (dips_loss_delta + dips_loss_delta_p)).item()
            # print(f"diff {diff_loss}")
            # print(f"dips :{dips_loss_delta + dips_loss_delta_p}")
            step_loss = diff_loss + self.dips_strength * (dips_loss_delta + dips_loss_delta_p)

            # temp_loss = self.dips_strength * (dips_loss_delta + dips_loss_delta_p)
            # temp_loss.backward()
            # for name, param in self.mod_adapter.named_parameters():
            #     print(f"{name} 的梯度: {param.grad}")
            # exit(0)
        else:
            step_loss = diff_loss
        assert not torch.isnan(step_loss).any(), "Loss contains NaN!"
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )

        return step_loss

    def generate_a_sample(self):
        raise NotImplementedError("Generate a sample not implemented.")


def main():
    # Initialize
    config = get_config()
    training_config = config["train"]

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    adapter_config_path = config["adapter_config_path"]
    adapter_config = get_config(config_path=adapter_config_path)
    # Initialize custom dataset
    dataset = CustomDataset(pkl_path=config["pkl_path"],
                            image_dir=config["data_path"],
                            user_num=training_config["user_num"],
                            json_path=config["json_path"])
    # Initialize model
    trainable_model = OminiModelDeltaVerse(
        model_path=config.get("model_path", None),
        flux_pipe_id=config["flux_path"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=adapter_config,
        is_dips=training_config["dips"],
        dips_stable=training_config["dips_stable"],
        user_num=training_config["user_num"],
        drop_user=training_config["dataset"]["drop_user_prob"],
        drop_text=training_config["dataset"]["drop_text_prob"],
    )

    train(dataset, trainable_model, config, adapter_config=adapter_config,test_function=empty_test_function)


if __name__ == "__main__":
    main()
