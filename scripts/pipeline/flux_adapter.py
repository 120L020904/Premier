# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
from typing import List, Union, Optional, Tuple, Dict, Any, Callable

from diffusers import FluxTransformer2DModel, FluxPipeline
from diffusers.models.attention_processor import Attention, F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import USE_PEFT_BACKEND, unscale_lora_layers, is_torch_version, scale_lora_layers, logger
from einops import rearrange
import math
from diffusers.models.embeddings import apply_rotary_emb

from scripts.pipeline.mod_adapters import CLIPModAdapter_2



def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                 scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    B = query.size(0)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(B, 1, L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        assert False
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight


def attn_forward(
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        condition_latents: torch.FloatTensor = None,
        text_cond_mask: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        cond_rotary_emb: Optional[torch.Tensor] = None,
        model_config: Optional[Dict[str, Any]] = {},
        store_attn_map: bool = False,
        latent_height: Optional[int] = None,
        timestep: Optional[torch.Tensor] = None,
        last_attn_map: Optional[torch.Tensor] = None,
        condition_sblora_weight: Optional[float] = None,
        latent_sblora_weight: Optional[float] = None,
) -> torch.FloatTensor:
    batch_size, _, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )

    is_sblock = encoder_hidden_states is None
    is_dblock = not is_sblock

    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(batch_size, -1, attn.heads,
                                                                                 head_dim).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads,
                                                                             head_dim).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(batch_size, -1, attn.heads,
                                                                                 head_dim).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    ####################################################################################################
    if store_attn_map and encoder_hidden_states is not None:
        seq_length = encoder_hidden_states_query_proj.shape[2]
        img_length = hidden_states.shape[1]
        hidden_states, attention_probs = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # (B, 24, S+HW, S+HW) -> (B, 24, HW, S)
        t2i_attention_probs = attention_probs[:, :, seq_length:seq_length + img_length, :seq_length]
        # (B, 24, S+HW, S+HW) -> (B, 24, S, HW) -> (B, 24, HW, S)
        i2t_attention_probs = attention_probs[:, :, :seq_length, seq_length:seq_length + img_length].transpose(-1, -2)

        if not hasattr(attn, "attn_maps"):
            attn.attn_maps = []
            attn.timestep = []

        attn.attn_maps.append(
            (
                rearrange(t2i_attention_probs, 'B attn_head (H W) attn_dim -> B attn_head H W attn_dim',
                          H=latent_height),
                rearrange(i2t_attention_probs, 'B attn_head (H W) attn_dim -> B attn_head H W attn_dim',
                          H=latent_height),
            )
        )

        attn.timestep.append(timestep.cpu())
        has_nan = torch.isnan(hidden_states).any().item()
        if has_nan:
            print("[attn_forward] detect nan hidden_states in store_attn_map")
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        has_nan = torch.isnan(hidden_states).any().item()
        if has_nan:
            print("[attn_forward] detect nan hidden_states")
    ####################################################################################################
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)

    if encoder_hidden_states is not None:

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1]:],
        )

        hidden_states = attn.to_out[0](hidden_states)  # linear proj
        hidden_states = attn.to_out[1](hidden_states)  # dropout
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return (
            (hidden_states, encoder_hidden_states, condition_latents)
            if condition_latents is not None
            else (hidden_states, encoder_hidden_states)
        )
    else:
        return hidden_states


def set_delta_by_start_end(
        start_ends,
        src_delta_emb, src_delta_emb_pblock,
        delta_emb, delta_emb_pblock, delta_emb_mask,
):
    for (i, j, src_s, src_e, tar_s, tar_e) in start_ends:
        if src_delta_emb is not None:
            delta_emb[i, tar_s:tar_e] = src_delta_emb[j, src_s:src_e]
        if src_delta_emb_pblock is not None:
            delta_emb_pblock[i, tar_s:tar_e] = src_delta_emb_pblock[j, src_s:src_e]
        delta_emb_mask[i, tar_s:tar_e] = True
    return delta_emb, delta_emb_pblock, delta_emb_mask


def norm1_context_forward(
        norm1_context,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
        delta_emb: Optional[torch.Tensor] = None,
        delta_emb_cblock: Optional[torch.Tensor] = None,
        mod_adapter=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_length = x.shape[:2]

    if mod_adapter is not None:
        assert False

    if delta_emb is None and delta_emb_cblock is None:
        emb = norm1_context.linear(norm1_context.silu(emb))  # (B, 3072) -> (B, 18432)
        emb = emb.unsqueeze(1)  # (B, 1, 18432)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)  # (B, 1, 3072)
        x = norm1_context.norm(x) * (1 + scale_msa) + shift_msa  # (B, 1, 3072)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    else:
        # (B, 3072) > (B, 18432) -> (B, S, 18432)
        # (B, 3072) -> (B, 1, 3072) -> (B, S, 3072) -> (B, S, 18432)
        if delta_emb_cblock is None:
            emb_new = norm1_context.linear(norm1_context.silu(emb.unsqueeze(1) + delta_emb))
        elif delta_emb is None:
            emb_new = norm1_context.linear(norm1_context.silu(emb.unsqueeze(1) + delta_emb_cblock))
        else:
            emb_new = norm1_context.linear(norm1_context.silu(emb.unsqueeze(1) + delta_emb + delta_emb_cblock))
        emb = emb_new
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)  # (B, S, 3072)
        x = norm1_context.norm(x) * (1 + scale_msa) + shift_msa  # (B, S, 3072)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


def norm1_forward(
        norm1,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
        delta_emb: Optional[torch.Tensor] = None,
        delta_emb_cblock: Optional[torch.Tensor] = None,
        delta_emb_mask: Optional[torch.Tensor] = None,
        t2i_attn_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if delta_emb is None:
        emb = norm1.linear(norm1.silu(emb))  # (B, 3072) -> (B, 18432)
        emb = emb.unsqueeze(1)  # (B, 1, 18432)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)  # (B, 1, 3072)
        x = norm1.norm(x) * (1 + scale_msa) + shift_msa  # (B, 1, 3072)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    else:
        raise NotImplementedError()



def block_forward(
        block,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        condition_latents: torch.FloatTensor,
        temb: torch.FloatTensor,
        cond_temb: torch.FloatTensor,
        text_cond_mask: Optional[torch.FloatTensor] = None,
        delta_emb: Optional[torch.FloatTensor] = None,
        delta_emb_cblock: Optional[torch.FloatTensor] = None,
        delta_start_ends=None,
        cond_rotary_emb=None,
        image_rotary_emb=None,
        model_config: Optional[Dict[str, Any]] = {},
        store_attn_map: bool = False,
        use_text_mod: bool = True,
        use_img_mod: bool = False,
        mod_adapter=None,
        latent_height: Optional[int] = None,
        timestep: Optional[torch.Tensor] = None,
        last_attn_map: Optional[torch.Tensor] = None,
):
    batch_size = hidden_states.shape[0]
    use_cond = condition_latents is not None

    train_partial_latent_lora = model_config.get("train_partial_latent_lora", False)
    train_partial_text_lora = model_config.get("train_partial_text_lora", False)
    if train_partial_latent_lora:
        train_partial_latent_lora_layers = model_config.get("train_partial_latent_lora_layers", "")
        activate_norm1 = activate_ff = True
        if "norm1" not in train_partial_latent_lora_layers:
            activate_norm1 = False
        if "ff" not in train_partial_latent_lora_layers:
            activate_ff = False

    if train_partial_text_lora:
        train_partial_text_lora_layers = model_config.get("train_partial_text_lora_layers", "")
        activate_norm1_context = activate_ff_context = True
        if "norm1" not in train_partial_text_lora_layers:
            activate_norm1_context = False
        if "ff" not in train_partial_text_lora_layers:
            activate_ff_context = False

    if use_cond:
        norm_condition_latents, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = (
            norm1_forward(
                block.norm1,
                condition_latents,
                emb=cond_temb,
            )
        )
    delta_emb_img = delta_emb_img_cblock = None
    if use_img_mod and use_text_mod:
        if delta_emb is not None:
            delta_emb_img, delta_emb = delta_emb.chunk(2, dim=-1)
        if delta_emb_cblock is not None:
            delta_emb_img_cblock, delta_emb_cblock = delta_emb_cblock.chunk(2, dim=-1)

    if use_img_mod and encoder_hidden_states is not None:
        with torch.no_grad():
            attn = block.attn

            norm_img = block.norm1(hidden_states, emb=temb)[0]
            norm_text = block.norm1_context(encoder_hidden_states, emb=temb)[0]

            img_query = attn.to_q(norm_img)
            img_key = attn.to_k(norm_img)
            text_query = attn.add_q_proj(norm_text)
            text_key = attn.add_k_proj(norm_text)

            inner_dim = img_key.shape[-1]
            head_dim = inner_dim // attn.heads

            img_query = img_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # (B, N, HW, D)
            img_key = img_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # (B, N, HW, D)
            text_query = text_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # (B, N, S, D)
            text_key = text_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # (B, N, S, D)

            if attn.norm_q is not None:
                img_query = attn.norm_q(img_query)
            if attn.norm_added_q is not None:
                text_query = attn.norm_added_q(text_query)
            if attn.norm_k is not None:
                img_key = attn.norm_k(img_key)
            if attn.norm_added_k is not None:
                text_key = attn.norm_added_k(text_key)

            query = torch.cat([text_query, img_query], dim=2)  # (B, N, S+HW, D)
            key = torch.cat([text_key, img_key], dim=2)  # (B, N, S+HW, D)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

            seq_length = text_query.shape[2]

            scale_factor = 1 / math.sqrt(query.size(-1))
            t2i_attn_map = query @ key.transpose(-2, -1) * scale_factor  # (B, N, S+HW, S+HW)
            t2i_attn_map = t2i_attn_map.mean(1)[:, seq_length:, :seq_length]  # (B, S+HW, S+HW) -> (B, HW, S)
            t2i_attn_map = torch.softmax(t2i_attn_map, dim=-1)  # (B, HW, S)

    else:
        t2i_attn_map = None

    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        norm1_forward(
            block.norm1,
            hidden_states,
            emb=temb,
            delta_emb=delta_emb_img,
            delta_emb_cblock=delta_emb_img_cblock,
            t2i_attn_map=t2i_attn_map,
        )
    )
    # Modulation for double block
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        norm1_context_forward(
            block.norm1_context,
            encoder_hidden_states,
            emb=temb,
            delta_emb=delta_emb if use_text_mod else None,
            delta_emb_cblock=delta_emb_cblock if use_text_mod else None,
            mod_adapter=mod_adapter,
        )
    )

    # Attention.
    result = attn_forward(
        block.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        condition_latents=norm_condition_latents if use_cond else None,
        text_cond_mask=text_cond_mask if use_cond else None,
        image_rotary_emb=image_rotary_emb,
        cond_rotary_emb=cond_rotary_emb if use_cond else None,
        store_attn_map=store_attn_map,
        latent_height=latent_height,
        timestep=timestep,
        last_attn_map=last_attn_map,
    )
    attn_output, context_attn_output = result[:2]
    cond_attn_output = result[2] if use_cond else None

    # Process attention outputs for the `hidden_states`.
    # 1. hidden_states
    attn_output = gate_msa * attn_output  # NOTE: changed by img mod
    hidden_states = hidden_states + attn_output
    # 2. encoder_hidden_states
    context_attn_output = c_gate_msa * context_attn_output  # NOTE: changed by delta_temb
    encoder_hidden_states = encoder_hidden_states + context_attn_output
    # 3. condition_latents
    if use_cond:
        cond_attn_output = cond_gate_msa * cond_attn_output  # NOTE: changed by img mod
        condition_latents = condition_latents + cond_attn_output
        if model_config.get("add_cond_attn", False):
            hidden_states += cond_attn_output

    # LayerNorm + MLP.
    # 1. hidden_states
    norm_hidden_states = block.norm2(hidden_states)
    norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp) + shift_mlp  # NOTE: changed by img mod
    )
    # 2. encoder_hidden_states
    norm_encoder_hidden_states = block.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp  # NOTE: changed by delta_temb
    )
    # 3. condition_latents
    if use_cond:
        norm_condition_latents = block.norm2(condition_latents)
        norm_condition_latents = (
                norm_condition_latents * (1 + cond_scale_mlp) + cond_shift_mlp  # NOTE: changed by img mod
        )

    # Feed-forward.
    # 1. hidden_states
    ff_output = block.ff(norm_hidden_states)
    ff_output = gate_mlp * ff_output  # NOTE: changed by img mod
    # 2. encoder_hidden_states
    context_ff_output = block.ff_context(norm_encoder_hidden_states)
    context_ff_output = c_gate_mlp * context_ff_output  # NOTE: changed by delta_temb
    # 3. condition_latents
    if use_cond:
        cond_ff_output = block.ff(norm_condition_latents)
        cond_ff_output = cond_gate_mlp * cond_ff_output  # NOTE: changed by img mod

    # Process feed-forward outputs.
    hidden_states = hidden_states + ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output
    if use_cond:
        condition_latents = condition_latents + cond_ff_output

    # Clip to avoid overflow.
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states, condition_latents if use_cond else None


def single_norm_forward(
        block,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
        delta_emb: Optional[torch.Tensor] = None,
        delta_emb_cblock: Optional[torch.Tensor] = None,
        text_seq_length: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    if delta_emb is None and delta_emb_cblock is None:
        emb = block.linear(block.silu(emb))  # (B, 3072) -> (B, 9216)
        emb = emb.unsqueeze(1)  # (B, 1, 9216)
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=-1)  # (B, 1, 3072)
        x = block.norm(x) * (1 + scale_msa) + shift_msa  # (B, S, 3072) * (B, 1, 3072)
        return x, gate_msa
    else:
        img_text_seq_length = x.shape[1]  # S+

        # (B, 3072) -> (B, 9216) -> (B, S+, 9216)
        emb_orig = block.linear(block.silu(emb)).unsqueeze(1).expand((-1, img_text_seq_length, -1))
        # (B, 3072) -> (B, 1, 3072) -> (B, S, 3072) -> (B, S, 9216)
        if delta_emb_cblock is None:
            emb_new = block.linear(block.silu(emb.unsqueeze(1) + delta_emb))
        elif delta_emb is None:
            emb_new = block.linear(block.silu(emb.unsqueeze(1) + delta_emb_cblock))
        else:
            emb_new = block.linear(block.silu(emb.unsqueeze(1) + delta_emb + delta_emb_cblock))

        emb_text = emb_new
        emb_img = emb_orig[:, text_seq_length:]  # (B, s, 9216)
        emb = torch.cat([emb_text, emb_img], dim=1)  # (B, S+, 9216)

        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=-1)  # (B, S+, 3072)
        x = block.norm(x) * (1 + scale_msa) + shift_msa  # (B, S+, 3072)
        return x, gate_msa


def single_block_forward(
        block,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        condition_latents: torch.FloatTensor = None,
        text_cond_mask: torch.FloatTensor = None,
        cond_temb: torch.FloatTensor = None,
        delta_emb: Optional[torch.FloatTensor] = None,
        delta_emb_cblock: Optional[torch.FloatTensor] = None,
        use_text_mod: bool = True,
        cond_rotary_emb=None,
        latent_height: Optional[int] = None,
        timestep: Optional[torch.Tensor] = None,
        store_attn_map: bool = False,
        model_config: Optional[Dict[str, Any]] = {},
        last_attn_map: Optional[torch.Tensor] = None,
        latent_sblora_weight=None,
        condition_sblora_weight=None,
):
    using_cond = condition_latents is not None
    residual = hidden_states

    train_partial_lora = model_config.get("train_partial_lora", False)
    if train_partial_lora:
        train_partial_lora_layers = model_config.get("train_partial_lora_layers", "")
        activate_norm = activate_projmlp = activate_projout = True

        if "norm" not in train_partial_lora_layers:
            activate_norm = False
        if "projmlp" not in train_partial_lora_layers:
            activate_projmlp = False
        if "projout" not in train_partial_lora_layers:
            activate_projout = False

    # Modulation for single block
    norm_hidden_states, gate = single_norm_forward(
        block=block.norm,
        x=hidden_states,
        emb=temb,
        delta_emb=delta_emb if use_text_mod else None,
        delta_emb_cblock=delta_emb_cblock if use_text_mod else None,
    )
    mlp_hidden_states = block.act_mlp(block.proj_mlp(norm_hidden_states))
    if using_cond:
        residual_cond = condition_latents
        norm_condition_latents, cond_gate = block.norm(condition_latents, emb=cond_temb)
        mlp_cond_hidden_states = block.act_mlp(block.proj_mlp(norm_condition_latents))

    attn_output = attn_forward(
        block.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        last_attn_map=last_attn_map,
        latent_height=latent_height,
        store_attn_map=store_attn_map,
        timestep=timestep,
        latent_sblora_weight=latent_sblora_weight,
        condition_sblora_weight=condition_sblora_weight,
        **(
            {
                "condition_latents": norm_condition_latents,
                "cond_rotary_emb": cond_rotary_emb if using_cond else None,
                "text_cond_mask": text_cond_mask if using_cond else None,
            }
            if using_cond
            else {}
        ),
    )
    if using_cond:
        attn_output, cond_attn_output = attn_output

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    # gate = (B, 1, 3072) or (B, S+, 3072)
    hidden_states = gate * block.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    if using_cond:
        condition_latents = torch.cat([cond_attn_output, mlp_cond_hidden_states], dim=2)
        cond_gate = cond_gate.unsqueeze(1)
        condition_latents = cond_gate * block.proj_out(condition_latents)
        condition_latents = residual_cond + condition_latents

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states if not using_cond else (hidden_states, condition_latents)


def prepare_params(
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        **kwargs: dict,
):
    return (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_single_block_samples,
        return_dict,
    )


def transformer_forward_verse(
        transformer: FluxTransformer2DModel,
        condition_latents: torch.Tensor =None,
        condition_ids: torch.Tensor = None,
        model_config: Optional[Dict[str, Any]] = {},
        c_t=0,
        text_cond_mask: Optional[torch.FloatTensor] = None,
        delta_emb: Optional[torch.FloatTensor] = None,
        delta_emb_pblock: Optional[torch.FloatTensor] = None,
        delta_start_ends=None,
        store_attn_map: bool = False,
        use_text_mod: bool = True,
        use_img_mod: bool = False,
        mod_adapter=None,
        latent_height: Optional[int] = None,
        last_attn_map=None,
        **params: dict,
):
    use_condition = condition_latents is not None

    (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_single_block_samples,
        return_dict,
    ) = prepare_params(**params)

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        latent_sblora_weight = joint_attention_kwargs.pop("latent_sblora_weight", None)
        condition_sblora_weight = joint_attention_kwargs.pop("condition_sblora_weight", None)
    else:
        lora_scale = 1.0
        latent_sblora_weight = None
        condition_sblora_weight = None
    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(transformer, lora_scale)
    else:
        if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    train_partial_text_lora = model_config.get("train_partial_text_lora", False)
    train_partial_latent_lora = model_config.get("train_partial_latent_lora", False)

    if train_partial_text_lora or train_partial_latent_lora:
        train_partial_text_lora_layers = model_config.get("train_partial_text_lora_layers", "")
        train_partial_latent_lora_layers = model_config.get("train_partial_latent_lora_layers", "")
        activate_x_embedder = True
        if "x_embedder" not in train_partial_text_lora_layers or "x_embedder" not in train_partial_latent_lora_layers:
            activate_x_embedder = False

    hidden_states = transformer.x_embedder(hidden_states)
    condition_latents = transformer.x_embedder(condition_latents) if use_condition else None

    timestep = timestep.to(hidden_states.dtype) * 1000

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        transformer.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else transformer.time_text_embed(timestep, guidance, pooled_projections)
    )  # (B, 3072)

    cond_temb = (
        transformer.time_text_embed(torch.ones_like(timestep) * c_t * 1000, pooled_projections)
        if guidance is None
        else transformer.time_text_embed(
            torch.ones_like(timestep) * c_t * 1000, guidance, pooled_projections
        )
    )
    encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = transformer.pos_embed(ids)
    if use_condition:
        cond_rotary_emb = transformer.pos_embed(condition_ids)

    for index_block, block in enumerate(transformer.transformer_blocks):
        if delta_emb_pblock is None:
            delta_emb_cblock = None
        else:
            delta_emb_cblock = delta_emb_pblock[:, :, index_block]
        condition_pass_to_double = use_condition and (
                model_config["double_use_condition"] or model_config["single_use_condition"])
        if transformer.training and transformer.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            encoder_hidden_states, hidden_states, condition_latents = (
                torch.utils.checkpoint.checkpoint(
                    block_forward,
                    block=block,
                    model_config=model_config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    condition_latents=condition_latents if condition_pass_to_double else None,
                    cond_temb=cond_temb if condition_pass_to_double else None,
                    cond_rotary_emb=cond_rotary_emb if condition_pass_to_double else None,
                    temb=temb,
                    text_cond_mask=text_cond_mask,
                    delta_emb=delta_emb,
                    delta_emb_cblock=delta_emb_cblock,
                    delta_start_ends=delta_start_ends,
                    image_rotary_emb=image_rotary_emb,
                    store_attn_map=store_attn_map,
                    use_text_mod=use_text_mod,
                    use_img_mod=use_img_mod,
                    mod_adapter=mod_adapter,
                    latent_height=latent_height,
                    timestep=timestep,
                    last_attn_map=last_attn_map,
                    **ckpt_kwargs,
                )
            )

        else:
            encoder_hidden_states, hidden_states, condition_latents = block_forward(
                block,
                model_config=model_config,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                condition_latents=condition_latents if condition_pass_to_double else None,
                cond_temb=cond_temb if condition_pass_to_double else None,
                cond_rotary_emb=cond_rotary_emb if condition_pass_to_double else None,
                temb=temb,
                text_cond_mask=text_cond_mask,
                delta_emb=delta_emb,
                delta_emb_cblock=delta_emb_cblock,
                delta_start_ends=delta_start_ends,
                image_rotary_emb=image_rotary_emb,
                store_attn_map=store_attn_map,
                use_text_mod=use_text_mod,
                use_img_mod=use_img_mod,
                mod_adapter=mod_adapter,
                latent_height=latent_height,
                timestep=timestep,
                last_attn_map=last_attn_map,
            )

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(transformer.transformer_blocks) / len(
                controlnet_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states = (
                    hidden_states
                    + controlnet_block_samples[index_block // interval_control]
            )
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(transformer.single_transformer_blocks):
        if delta_emb_pblock is not None and delta_emb_pblock.shape[2] > 19 + index_block:
            delta_emb_single = delta_emb
            delta_emb_cblock = delta_emb_pblock[:, :, index_block + 19]
        else:
            delta_emb_single = None
            delta_emb_cblock = None
        if transformer.training and transformer.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            result = torch.utils.checkpoint.checkpoint(
                single_block_forward,
                block=block,
                model_config=model_config,
                hidden_states=hidden_states,
                temb=temb,
                delta_emb=delta_emb_single,
                delta_emb_cblock=delta_emb_cblock,
                use_text_mod=use_text_mod,
                use_img_mod=use_img_mod,
                image_rotary_emb=image_rotary_emb,
                last_attn_map=last_attn_map,
                latent_height=latent_height,
                timestep=timestep,
                store_attn_map=store_attn_map,
                **(
                    {
                        "condition_latents": condition_latents,
                        "cond_temb": cond_temb,
                        "cond_rotary_emb": cond_rotary_emb,
                        "text_cond_mask": text_cond_mask,
                    }
                    if use_condition and model_config["single_use_condition"]
                    else {}
                ),
                **ckpt_kwargs,
            )

        else:
            result = single_block_forward(
                block,
                model_config=model_config,
                hidden_states=hidden_states,
                temb=temb,
                delta_emb=delta_emb_single,
                delta_emb_cblock=delta_emb_cblock,
                use_text_mod=use_text_mod,
                image_rotary_emb=image_rotary_emb,
                last_attn_map=last_attn_map,
                latent_height=latent_height,
                timestep=timestep,
                store_attn_map=store_attn_map,
                latent_sblora_weight=latent_sblora_weight,
                condition_sblora_weight=condition_sblora_weight,
                **(
                    {
                        "condition_latents": condition_latents,
                        "cond_temb": cond_temb,
                        "cond_rotary_emb": cond_rotary_emb,
                        "text_cond_mask": text_cond_mask,
                    }
                    if use_condition and model_config["single_use_condition"]
                    else {}
                ),
            )
        if use_condition and model_config["single_use_condition"]:
            hidden_states, condition_latents = result
        else:
            hidden_states = result

        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(transformer.single_transformer_blocks) / len(
                controlnet_single_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
            )

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

    hidden_states = transformer.norm_out(hidden_states, temb)
    output = transformer.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(transformer, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
@torch.no_grad()
def generate_xverse(
        pipeline: FluxPipeline,
        user_preference_embedding: torch.Tensor = None,
        mod_adapter: CLIPModAdapter_2 = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # Condition Parameters (Optional)
        image_guidance_scale: float = 1.0,
        transformer_kwargs: Optional[Dict[str, Any]] = {},
        kv_cache=False,
        model_config={},
        latent_mask=None,
        is_ori=True,
        **params: dict,
):
    self = pipeline

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    is_uncond = model_config["model"]["modulation"].get("uncond", False)
    # Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs

    # Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # Prepare prompt embeddings
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    # Prepare prompt embeddings
    (
        uncond_embeds,
        _,
        _,
    ) = self.encode_prompt(
        prompt="",
        prompt_2="",
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    # Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    if latent_mask is not None:
        latent_mask = latent_mask.T.reshape(-1)
        latents = latents[:, latent_mask]
        latent_image_ids = latent_image_ids[latent_mask]


    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)

    # Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000

            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance, c_guidances = None, [None for _ in c_guidances]
            # print(redux_pipeline is not None)
            if is_uncond:
                delta_emb, delta_emb_pblock = mod_adapter(t, prompt_embeds, user_preference_embedding, uncond_embeds)
            else:
                delta_emb, delta_emb_pblock = mod_adapter(t, prompt_embeds, user_preference_embedding)
            noise_pred = transformer_forward_verse(
                self.transformer,
                model_config=model_config,
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance,
                timestep=timestep,
                return_dict=False,
                delta_emb=delta_emb,
                delta_emb_pblock=delta_emb_pblock,
                **transformer_kwargs,
            )[0]



            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
                          latents / self.vae.config.scaling_factor
                  ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)



