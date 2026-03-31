from comfy_api.latest import ComfyExtension, io

from .nodes import InfiniteTalkAutoSampler, InfiniteTalkAutoSamplerAdvanced

print(
    "🎵 [InfiniteTalk Native Sampler] Loaded: InfiniteTalkAutoSampler, InfiniteTalkAutoSamplerAdvanced"
)


class InfiniteTalkExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [InfiniteTalkAutoSampler, InfiniteTalkAutoSamplerAdvanced]


async def comfy_entrypoint() -> InfiniteTalkExtension:
    return InfiniteTalkExtension()
