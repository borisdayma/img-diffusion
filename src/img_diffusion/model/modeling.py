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
from transformers.modeling_flax_utils import ACT2FN
from transformers.utils import logging

from .configuration import ImgDiffusionConfig
from .utils import PretrainedFromWandbMixin

logger = logging.get_logger(__name__)


def timestep_embedding(timesteps, dim, max_period=10_000):
    SCALING_FACTOR = 5_000  # input scale is [0, 1]
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period)
        * jnp.arange(start=0, stop=half, step=1, dtype=jnp.float32)
        / half
    )

    args = timesteps[:None].astype(jnp.float32) * freqs[None, :] * SCALING_FACTOR
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
    num_heads: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, encoder_outputs=None, deterministic: bool = True):
        norm = partial(nn.LayerNorm, dtype=self.dtype, use_scale=False)
        x = norm()(x)

        # flatten
        if x.ndim == 4:
            b, h, w, c = x.shape
            x = rearrange(x, "b h w c -> b (h w) c")

        # attention
        attention = partial(
            nn.attention.MultiHeadDotProductAttention,
            num_heads=self.num_heads,
            dtype=self.dtype,
            use_bias=self.config.use_bias,
            dropout_rate=self.config.attention_dropout,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        x = attention()(inputs_q=x, inputs_kv=x, deterministic=deterministic)

        # cross-attention
        if encoder_outputs is not None and self.config.use_cross_attention:
            x = attention()(
                inputs_q=x, inputs_kv=encoder_outputs, deterministic=deterministic
            )

        # reshape
        if x.ndim == 4:
            x = rearrange(x, "b (h w) c -> b h w c", h=h)

        # norm
        x = norm()(x)

        return x


class ResBlock(nn.Module):
    config: ImgDiffusionConfig
    channels: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, time_embeddings, deterministic: bool = True):
        conv = partial(
            nn.Conv,
            features=self.channels,
            kernel_size=(3, 3),
            strides=1,
            padding="SAME",
            use_bias=self.config.use_bias,
            dtype=self.dtype,
        )
        norm = partial(nn.GroupNorm, num_groups=32, dtype=self.dtype, use_scale=False)
        h = norm()(x)
        h = ACT2FN(self.config.activation_function)(h)
        h = conv()(h)
        time_embeddings = ACT2FN(self.config.activation_function)(time_embeddings)
        time_embeddings = nn.Dense(
            features=self.channels,
            use_bias=self.config.use_bias,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=self.dtype,
        )(time_embeddings)
        time_embeddings = rearrange(time_embeddings, "b c -> b 1 1 c")
        h = h + time_embeddings
        h = norm()(h)
        h = ACT2FN(self.config.activation_function)(h)
        h = nn.Dropout(rate=self.config.activation_dropout)(
            h, deterministic=deterministic
        )
        h = conv()(h)
        return x + h


class FFN(nn.Module):
    config: ImgDiffusionConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        norm = partial(nn.LayerNorm, dtype=self.dtype, use_scale=False)
        dense = partial(
            nn.Dense,
            self.config.text_ffn_dim,
            dtype=self.dtype,
            use_bias=self.config.use_bias,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        x = norm()(x)
        x = dense()(x)
        x = ACT2FN[self.config.activation_function](x)
        x = norm()(x)
        x = nn.Dropout(rate=self.config.activation_dropout)(
            x, deterministic=deterministic
        )
        x = dense()(x)
        x = nn.Dropout(rate=self.config.dropout)(x, deterministic=deterministic)
        return x


class GLU(nn.Module):
    config: ImgDiffusionConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        norm = partial(nn.LayerNorm, dtype=self.dtype, use_scale=False)
        dense = partial(
            nn.Dense,
            self.config.text_ffn_dim,
            dtype=self.dtype,
            use_bias=self.config.use_bias,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        x = norm()(x)
        w = dense()(x)
        w = ACT2FN[self.config.activation_function](w)
        v = dense()(x)
        x = w * v
        x = norm()(x)
        x = nn.Dropout(rate=self.config.activation_dropout)(
            x, deterministic=deterministic
        )
        x = dense()(x)
        x = nn.Dropout(rate=self.config.dropout)(x, deterministic=deterministic)
        return x


class TransformerLayer(nn.Module):
    config: ImgDiffusionConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, deterministic: bool = True):
        # attention
        x = AttentionBlock(
            config=self.config, num_heads=self.config.text_heads, dtype=self.dtype
        )(x, deterministic=deterministic)
        hidden_states = hidden_states + x

        # ffn
        if self.config.use_glu:
            x = GLU(config=self.config, dtype=self.dtype)(
                hidden_states, deterministic=deterministic
            )
        else:
            x = FFN(config=self.config, dtype=self.dtype)(
                hidden_states, deterministic=deterministic
            )
        hidden_states = hidden_states + x

        return (hidden_states, None)


class Transformer(nn.Module):
    config: ImgDiffusionConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True):
        input_shape = input_ids.shape
        assert input_shape[-1] == self.config.max_text_length
        input_ids = input_ids.reshape(-1, input_shape[-1])

        embed = partial(
            nn.Embed,
            features=self.config.text_dim,
            dtype=self.dtype,
            embedding_init=jax.nn.initializers.normal(self.config.init_std),
        )
        inputs_embeds = embed(
            num_embeddings=self.config.vocab_size,
        )(input_ids)

        batch_size, sequence_length = input_ids.shape
        position_ids = jnp.broadcast_to(
            jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
        )
        embed_pos = embed(num_embeddings=self.config.max_text_length)(position_ids)

        hidden_states = inputs_embeds + embed_pos

        norm = partial(nn.LayerNorm, dtype=self.dtype)

        hidden_states = norm(hidden_states, use_scale=True)
        hidden_states = nn.Dropout(rate=self.config.dropout)(
            hidden_states, deterministic=deterministic
        )

        # layers
        hidden_states = (hidden_states,)
        hidden_states, _ = nn.scan(
            TransformerLayer,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.config.text_layers,
        )(self.config, dtype=self.dtype, name="layers",)(
            hidden_states,
            deterministic,
        )
        hidden_states = hidden_states[0]

        # final norm
        hidden_states = norm(hidden_states, use_scale=False)

        return hidden_states


class ImgDiffusionModule(nn.Module):
    config: ImgDiffusionConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x,
        timesteps,
        deterministic: bool = True,
        img_inputs=None,
        text_inputs=None,
    ):
        b, h, w, c = x.shape

        if img_inputs is not None:
            # concatenate x with image input
            assert (
                x.shape[:2] == img_inputs.shape[:2]
            ), f"x and img_inputs must have same batch, width, height, but got {x.shape} and {img_inputs.shape}"

            x = jnp.concatenate([x, img_inputs], axis=-1)

        # time embedding
        time_embedding = TimeEmbedding(config=self.config, dtype=self.dtype)(timesteps)

        # compute text embeddings
        if text_inputs is not None:
            # TODO: avoid recomputing text embedding at each diffusion step (cache or separate functions)
            text_inputs = Transformer(config=self.config, dtype=self.dtype)(text_inputs)

            # add to time embedding
            text_c = text_inputs.shape[-1]
            time_c = time_embedding.shape[-1]
            if text_c != time_c:
                text_proj = nn.Dense(
                    time_c,
                    dtype=self.dtype,
                    use_bias=self.config.use_bias,
                    kernel_init=jax.nn.initializers.normal(self.config.init_std),
                )(text_inputs)
            time_embedding += text_proj

        # U-net
        hidden_states = []
        for layer, (ch_mult, attention_block) in enumerate(
            zip(self.config.channel_mult, self.config.attention_block)
        ):
            channels = self.config.model_channels * ch_mult
            for idx in range(self.config.blocks_per_layer):
                # TODO: we should scan when no attention layer
                x = ResBlock(
                    config=self.config,
                    channels=channels,
                    dtype=self.dtype,
                )(x, time_embedding, deterministic=deterministic)
                x = nn.Dropout(rate=self.config.activation_dropout)(
                    x, deterministic=deterministic
                )
                if attention_block and idx < self.config.blocks_per_layer - 1:
                    c = x.shape[-1]
                    x = AttentionBlock(
                        config=self.config,
                        num_heads=c // self.config.num_head_channels,
                        dtype=self.dtype,
                    )(x, encoder_outputs=text_inputs, deterministic=deterministic)
                    x = nn.Dropout(rate=self.config.activation_dropout)(
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
                x = nn.Dropout(rate=self.config.activation_dropout)(
                    x, deterministic=deterministic
                )

        for ch_mult, attention_block in reversed(
            zip(self.config.channel_mult[:-1], self.config.attention_block[:-1])
        ):
            channels = self.config.model_channels * ch_mult

            # upsample
            x = nn.ConvTranspose(
                features=channels,
                kernel_size=(3, 3),
                strides=2,
                padding=1,
                use_bias=self.config.use_bias,
                dtype=self.dtype,
            )(x)
            x = nn.Dropout(rate=self.config.activation_dropout)(
                x, deterministic=deterministic
            )

            # skip connection
            x = x + hidden_states.pop()

            for idx in range(self.config.blocks_per_layer):
                x = ResBlock(
                    config=self.config,
                    channels=channels,
                    dtype=self.dtype,
                )(x, time_embedding, deterministic=deterministic)
                x = nn.Dropout(rate=self.config.activation_dropout)(
                    x, deterministic=deterministic
                )
                if attention_block and idx < self.config.blocks_per_layer - 1:
                    c = x.shape[-1]
                    x = AttentionBlock(
                        config=self.config,
                        num_heads=c // self.config.num_head_channels,
                        dtype=self.dtype,
                    )(x, encoder_outputs=text_inputs, deterministic=deterministic)
                    x = nn.Dropout(rate=self.config.activation_dropout)(
                        x, deterministic=deterministic
                    )
        # final output
        x = nn.Conv(
            features=c,
            kernel_size=(1, 1),
            use_bias=self.config.use_bias,
            dtype=self.dtype,
        )(x)

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
