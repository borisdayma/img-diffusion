import math
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from flax.core.frozen_dict import unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import FlaxPreTrainedModel
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from transformers.utils import logging

from .configuration import ImgDiffusionConfig
from .utils import PretrainedFromWandbMixin

logger = logging.get_logger(__name__)


def timestep_embedding(timesteps, dim, max_period=10_000):
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period)
        * jnp.arange(start=0, stop=half, step=1, dtype=jnp.float32)
        / half
    )

    args = timesteps[:None].astype(jnp.float32) * freqs[None, :]
    embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate(
            [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
        )
    return embedding


class TimeEmbedding(nn.Module):
    config: ImgDiffusionConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, timesteps):
        time_embed_dim = self.config.model_channels * 4
        embeddings = timestep_embedding(timesteps, self.config.model_channels)
        dense = partial(
            nn.Dense,
            features=time_embed_dim,
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        embeddings = dense()(embeddings)
        embeddings = ACT2FN(self.config.activation_function)(embeddings)
        embeddings = dense()(embeddings)
        return embeddings


class AttentionBlock(nn.Module):
    config: ImgDiffusionConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        return x


class ConvNextBlock(nn.Module):
    config: ImgDiffusionConfig
    channels: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, time_embeddings):
        conv = partial(
            nn.Conv,
            features=self.channels,
            strides=1,
            padding="SAME",
            use_bias=self.config.use_bias,
            dtype=self.dtype,
        )
        norm = partial(nn.LayerNorm, dtype=self.dtype, use_scale=False)
        h = norm(x)
        h = ACT2FN(self.config.activation_function)(h)
        h = conv(
            kernel_size=(7, 7),
            feature_group_count=self.channels,
        )(h)
        time_embeddings = ACT2FN(self.config.activation_function)(time_embeddings)
        time_embeddings = nn.Dense(
            features=self.channels,
            use_bias=self.config.use_bias,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )(time_embeddings)
        time_embeddings = rearrange(time_embeddings, "b c -> b 1 1 c")
        h = h + time_embeddings
        h = nn.LayerNorm(dtype=self.dtype, use_scale=False)(h)
        h = ACT2FN(self.config.activation_function)(h)
        h = conv(kernel_size=(1, 1))(h)
        return x + h


class ImgDiffusionModule(nn.Module):
    config: ImgDiffusionConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, img_inputs, timesteps, deterministic: bool = True):
        # concatenate x with image input
        assert (
            x.shape == img_inputs.shape
        ), f"x and img_inputs must have the same shape, but got {x.shape} and {img_inputs.shape}"
        x = jnp.concatenate([x, img_inputs], axis=1)

        # time embedding
        time_embedding = TimeEmbedding(config=self.config, dtype=self.dtype)(timesteps)

        # U-net
        hidden_states = []
        for layer, (ch_mult, attention_block) in enumerate(
            zip(self.config.channel_mult, self.config.attention_block)
        ):
            channels = self.config.model_channels * ch_mult
            for _ in range(self.config.blocks_per_layer):
                x = ConvNextBlock(
                    config=self.config,
                    channels=channels,
                    dtype=self.dtype,
                )(x, time_embedding, deterministic=deterministic)
                if attention_block:
                    x = AttentionBlock(config=self.config, dtype=self.dtype)(
                        x, deterministic=deterministic
                    )

            if layer < len(self.config.channel_mult) - 1:
                # append to hidden states
                hidden_states.append(x)

                # downsample
                x = nn.Conv(
                    features=channels,
                    kernel_size=(3, 3),
                    strides=2,
                    padding=1,
                    use_bias=self.config.use_bias,
                    dtype=self.dtype,
                )(x)

        # mid blocks

        # upsample

        return x


class ImgDiffusion(PretrainedFromWandbMixin, FlaxPreTrainedModel):
    module_class = ImgDiffusionModule
    config_class = ImgDiffusionConfig

    def num_params(self, params=None):
        if params is None:
            params = self.params
        num_params = jax.tree_map(
            lambda param: param.size, flatten_dict(unfreeze(params))
        ).values()
        return sum(list(num_params))

    def unscan(self, params):
        if self.config.use_scan:
            self.config.use_scan = False
            params = flatten_dict(params)
            scanned_keys = [k for k in params.keys() if "layers" in k]
            for k in scanned_keys:
                v = params[k]
                name_idx = k.index("layers") + 1
                for i in range(len(v)):
                    new_k = (
                        *k[:name_idx],
                        f"{k[name_idx][:-1]}_{i}",
                        *k[name_idx + 1 :],
                    )
                    params[new_k] = v[i]
                del params[k]
            params = unflatten_dict(params)
        return params
