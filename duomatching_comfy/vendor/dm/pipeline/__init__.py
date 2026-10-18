"""ComfyUI-DuoMatching shim: upstream's version also imports the training pipeline."""
from .causal_inference import CausalInferencePipeline

__all__ = ["CausalInferencePipeline"]
