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
        use_bias=True,
        init_std=0.02,
        activation_function="gelu",
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
        self.use_bias = use_bias
        self.init_std = init_std
        self.activation_function = activation_function
        super().__init__(**kwargs)
