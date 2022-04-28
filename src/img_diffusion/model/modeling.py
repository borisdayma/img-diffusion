import math
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import FlaxPreTrainedModel
from transformers.modeling_flax_utils import FlaxPreTrainedModel
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
        embeddings = nn.silu()(embeddings)
        embeddings = dense()(embeddings)
        return embeddings


class ImgDiffusionModule(nn.Module):
    config: ImgDiffusionConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, img_inputs, timesteps, deterministic: bool = True):
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
