from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .utils import PretrainedFromWandbMixin

logger = logging.get_logger(__name__)


class ImgDiffusionConfig(PretrainedFromWandbMixin, PretrainedConfig):
    model_type = "imgdiffusion"

    def __init__(
        self,
        loss="l2",  # can be l1 or l2
        model_channels=192,
        use_bias=True,
        init_std=0.02,
        **kwargs,
    ):
        assert loss in ["l1", "l2"], "loss must be either l1 or l2"
        self.loss = loss
        self.model_channels = model_channels
        self.use_bias = use_bias
        self.init_std = init_std
        super().__init__(**kwargs)
