import gc
import json
import os
import sys
import time

import pandas as pd
from scripts.pipeline.flux_adapter import transformer_forward_verse
from scripts.train_flux.adapter_trainer import get_config, train
from torch import nn

sys.path.append("/home/nfs/nfs-27/wangzihao/dxm/wangzihao_dxm/OminiControl")
from torch.utils.data import Dataset
import torchvision.transforms as T

from scripts.utils.utils import get_image_dxm_csv
import os.path

import prodigyopt
import torch
from diffusers import FluxPipeline
from safetensors.torch import save_file, load_file
from scripts.pipeline.flux_omini import encode_images
from scripts.pipeline.mod_adapters import load_modulation_adapter
import lightning as L


@torch.no_grad()
def empty_test_function(model, save_path, file_name):
    return


class EmbeddingLinearCombination(nn.Module):
    def __init__(self, embedding_num, combination_size, use_softmax=True):
        """
        对所有Embedding进行可训练的线性组合

        Args:
            num_embeddings (int): 原始Embedding的数量（词汇表大小）[1,5](@ref)
            embedding_dim (int): 每个Embedding向量的维度[1,5](@ref)
            combination_size (int): 线性组合后新Embedding的数量
            use_softmax (bool): 是否对组合系数使用Softmax归一化，默认为True
        """
        super(EmbeddingLinearCombination, self).__init__()
        self.combination_size = combination_size
        self.use_softmax = use_softmax

        # 可训练的线性组合系数矩阵 [4](@ref)
        # 形状: (combination_size, num_embeddings)
        self.combination_weights = nn.Parameter(torch.randn(combination_size, embedding_num))

        # 初始化组合系数
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重参数"""
        # 使用Xavier初始化组合系数矩阵[4](@ref)
        nn.init.xavier_uniform_(self.combination_weights)
        # Embedding层的权重通常会使用默认的初始化方式，也可以自定义[4](@ref)

    def forward(self, user_embedding, input_ids=None):
        """
        前向传播

        Args:
            input_ids (torch.LongTensor, optional): 输入索引张量。如果为None，则返回整个组合后的Embedding矩阵

        Returns:
            torch.Tensor: 组合后的Embedding表示
        """
        # 获取原始Embedding权重矩阵 [1,5](@ref)
        # shape: (num_embeddings, embedding_dim)
        original_embedding_matrix = user_embedding.weight

        # 处理组合系数
        if self.use_softmax:
            # 使用Softmax确保每个组合的系数和为1，增加解释性
            combination_coeff = torch.softmax(self.combination_weights, dim=-1)
        else:
            combination_coeff = self.combination_weights

        # 进行线性组合: (combination_size, num_embeddings) × (num_embeddings, embedding_dim)
        # 结果: (combination_size, embedding_dim)
        combined_embedding_matrix = torch.matmul(combination_coeff, original_embedding_matrix)

        # 如果没有提供input_ids，直接返回组合后的Embedding矩阵
        if input_ids is None:
            return combined_embedding_matrix

        # 如果提供了input_ids，使用组合后的矩阵进行查找
        # input_ids shape: (batch_size, seq_len)
        # 返回: (batch_size, seq_len, embedding_dim)
        return combined_embedding_matrix[input_ids]

    def get_combination_weights(self):
        """获取当前组合系数，可用于分析或可视化"""
        if self.use_softmax:
            return torch.softmax(self.combination_weights, dim=-1)
        return self.combination_weights


class CustomDataset(Dataset):
    def __init__(
            self,
            condition_size=(512, 512),
            target_size=(512, 512),
            drop_text_prob: float = 0,
            drop_image_prob: float = 0,
            return_pil_image: bool = False,
            position_scale=1.0,
            train_df_path_format="../pickapic_dxm_coco_sparse/split/{}.csv",
            user_idx=-1,
            data_path = "",
    ):
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale
        train_df_path = train_df_path_format.format(user_idx)
        self.train_df = pd.read_csv(train_df_path)
        self.data_path = data_path
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        image_data = self.train_df.iloc[idx]

        images = get_image_dxm_csv([image_data], image_dir=self.data_path)

        image = images[0]

        image = image.resize(self.target_size).convert("RGB")
        description = image_data["caption"]
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
            softmax=False,
            gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
            flux_pipe_id, torch_dtype=dtype
        ).to(device)
        self.init_pipe()
        self.user_idx = user_idx
        # Initialize LoRA layers
        # self.model_config["model"]["modulation"]["uncond"] = True
        self.train_layers = self.init_mod_adapter(model_path=model_path,
                                                  config=model_config,
                                                  token_num=model_config["model"]["modulation"]["user_token_num"],
                                                  softmax=softmax,
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
                         user_num: int = 1000,
                         token_num: int = 30,
                         softmax: bool = False):
        self.mod_adapter = load_modulation_adapter(config, dtype, device, ckpt_dir=model_path, is_training=True)
        self.mod_adapter.requires_grad_(False).eval()
        self.token_num = token_num
        self.user_embedding = nn.Embedding(num_embeddings=user_num,
                                           embedding_dim=token_num * 1024,
                                           ).to(device=device, dtype=dtype)
        state_dict = load_file(os.path.join(model_path, "user_embedding.safetensors"))
        self.user_embedding.load_state_dict(state_dict)
        self.embeddingCombination = EmbeddingLinearCombination(combination_size=1, embedding_num=user_num,
                                                               use_softmax=softmax)

        # self.user_embedding_param.grad = torch.zeros_like(self.user_embedding_param.data)  # 初始化为零
        self.embeddingCombination.train()
        self.user_embedding.requires_grad_(False).eval()
        self.embeddingCombination.combination_weights.requires_grad_(True)
        if self.optimizer_config["type"] == "AdamW":
            param_list = [{"params": list(self.embeddingCombination.parameters()), "lr": 0.01}]
        else:
            param_list = [{"params": list(self.embeddingCombination.parameters())}]
        return param_list

    def save_lora(self, path: str):
        os.makedirs(path, exist_ok=True)

        user_state_dict = self.embeddingCombination.state_dict()
        save_file(
            tensors=user_state_dict,
            filename=os.path.join(path, f"user_combination_{self.user_idx}.safetensors"),
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
        user_embedding_ = self.embeddingCombination(input_ids=torch.tensor(len(prompts) * [0]),
                                                    user_embedding=self.user_embedding)
        user_preference_embedding = user_embedding_.view(-1, self.token_num, 1024)
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
                prompt_2=None,
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
    with open("../pickapic_dxm_coco/json/user.json") as f:
        user_id_list = json.load(f)
    user_index_list = user_id_list[3030:3040]
    model_name = "20250829-215051"
    experient_name = "xverse_test2_cross_dips_ori"
    step = 260000 # checkpoint step for loading the adapter model, which should be consistent with the checkpoint step used for training the adapter model. For example, if the adapter model is trained for 260k steps and the checkpoint is saved at 260k steps, then this step should be set to 260000. If the checkpoint is saved at 300k steps, then this step should be set to 300000.
    save_path = f"/runs/{experient_name}/{model_name}" # adapter model save path, which should contain the user_embedding.safetensors and the modulation adapter checkpoint (e.g., adapter.pth or ckpt/xxx.ckpt)
    model_save_path = f"{save_path}/ckpt/{step}"
    config_path = f"../OminiControl/train/config/personalize_xverse_user_linear.yaml"

    config = get_config(config_path)
    config["adapter_path"] = save_path
    training_config = config["train"]
    adapter_config_path = f"{save_path}/adapter_config.yaml"
    adapter_config = get_config(config_path=adapter_config_path)
    adapter_config["model"]["modulation"]["uncond"] = False
    run_name = time.strftime("%Y%m%d-%H%M%S")
    # run_name = "20250911-125016"
    print(run_name)
    for user_idx in user_index_list:
        # Initialize custom dataset
        print(f"user_idx {user_idx}")
        print(f"model path {model_save_path}")
        # if user_idx <= 18096:
        #     continue
        dataset = CustomDataset(user_idx=user_idx, train_df_path_format=config["csv_path"], data_path=config["data_path"])
        trainable_model = OminiModelUserEmbedding(
            model_path=model_save_path,
            flux_pipe_id=config["flux_path"],
            device=f"cuda",
            dtype=getattr(torch, config["dtype"]),
            optimizer_config=training_config["optimizer"],
            model_config=adapter_config,
            gradient_checkpointing=training_config.get("gradient_checkpointing", False),
            user_idx=user_idx,
            softmax=training_config.get("softmax", False)
        )
        train(dataset, trainable_model, config, adapter_config=adapter_config, test_function=empty_test_function,
              run_name=run_name)
        del trainable_model
        cuda_clear()


if __name__ == "__main__":
    main()
