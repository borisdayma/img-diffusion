from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .utils import PretrainedFromWandbMixin

logger = logging.get_logger(__name__)


class ImgDiffusionConfig(PretrainedFromWandbMixin, PretrainedConfig):
    model_type = "imgdiffusion"

    def __init__(
        self,
        loss="l2",  # can be l1 or l2
        model_channels=32,
        channel_mult=(1, 2, 4, 8),
        attention_block=(False, False, False, True),
        blocks_per_layer=2,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        num_head_channels=64,
        use_bias=False,
        init_std=0.02,
        activation_function="gelu",
        vocab_size=50264,
        text_dim=512,
        text_heads=8,
        text_ffn_dim=1024,
        text_layers=12,
        max_text_length=64,
        use_cross_attention=False,
        **kwargs,
    ):
        assert loss in ["l1", "l2"], "loss must be either l1 or l2"
        self.loss = loss
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.attention_block = attention_block
        assert len(channel_mult) == len(
            attention_block
        ), f"channel_mult and attention_block must have the same length, but got {len(channel_mult)} and {len(attention_block)}"
        self.blocks_per_layer = blocks_per_layer
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.num_head_channels = num_head_channels
        self.use_bias = use_bias
        self.init_std = init_std
        self.activation_function = activation_function
        self.vocab_size = vocab_size
        self.text_dim = text_dim
        self.text_heads = text_heads
        self.text_ffn_dim = text_ffn_dim
        self.text_layers = text_layers
        self.max_text_length = max_text_length
        self.use_cross_attention = use_cross_attention
        super().__init__(**kwargs)
