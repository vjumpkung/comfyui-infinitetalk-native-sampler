"""
Temporary patches for ComfyUI core compatibility issues.
Applied at import time so the core files remain unmodified.
"""
import logging


def _patch_wav2vec2_dtype():
    """
    Fix: RuntimeError: Input type (float) and bias type (c10::Half) should be the same
    The PositionalConvEmbedding.forward() passes float32 input to a fp16 conv.
    Cast input to match the conv weight dtype before the operation.
    """
    try:
        from comfy.audio_encoders import wav2vec2

        PositionalConvEmbedding = wav2vec2.PositionalConvEmbedding
        original_forward = PositionalConvEmbedding.forward

        def patched_forward(self, x):
            x = x.transpose(1, 2)
            x = self.conv(x.to(self.conv.weight.dtype))[:, :, :-1]
            x = self.activation(x)
            x = x.transpose(1, 2)
            return x

        PositionalConvEmbedding.forward = patched_forward
        logging.info(
            "[InfiniteTalk Native Sampler] Applied wav2vec2 dtype patch."
        )
    except Exception as e:
        logging.warning(
            f"[InfiniteTalk Native Sampler] Could not apply wav2vec2 dtype patch: {e}"
        )


_patch_wav2vec2_dtype()
