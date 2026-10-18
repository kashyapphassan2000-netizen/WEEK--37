"""awq-from-scratch: universal, block-streaming AWQ 4-bit quantization in pure PyTorch."""
from .awq_linear import AwqExpertsWeight, AwqLinear
from .config import RunConfig, load_config
from .model_io import load_quantized

__all__ = ["AwqLinear", "AwqExpertsWeight", "RunConfig", "load_config", "load_quantized"]
__version__ = "1.0.0"
