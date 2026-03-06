from .nodes import InfiniteTalkAutoSampler, InfiniteTalkAutoSamplerAdvanced
from . import _patches  # noqa: F401

# Module import message
print(
    "🎵 [InfiniteTalk Native Sampler] Loaded: InfiniteTalkAutoSampler, InfiniteTalkAutoSamplerAdvanced"
)

NODE_CLASS_MAPPINGS = {
    "InfiniteTalkAutoSampler": InfiniteTalkAutoSampler,
    "InfiniteTalkAutoSamplerAdvanced": InfiniteTalkAutoSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InfiniteTalkAutoSampler": "InfiniteTalk Auto Sampler",
    "InfiniteTalkAutoSamplerAdvanced": "InfiniteTalk Auto Sampler (Advanced)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
