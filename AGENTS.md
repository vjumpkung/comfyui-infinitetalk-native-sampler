# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

ComfyUI custom node package for InfiniteTalk - generates lip-sync videos from audio using WAN 2.1 models with automatic frame extension for long audio.

## Architecture Patterns

### ComfyUI Node Structure
- Nodes defined in [`nodes.py`](nodes.py) as classes with `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY` class attributes
- Custom ComfyUI type annotations: `"MODEL"`, `"MODEL_PATCH"`, `"CONDITIONING"`, `"AUDIO_ENCODER_OUTPUT"`, `"NOISE"`, `"SAMPLER"`, `"SIGMAS"`
- Node mappings exported from [`__init__.py`](__init__.py:8) via `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`

### Model Patching System
- Uses [`comfy.patcher_extension.WrappersMP.OUTER_SAMPLE`](nodes.py:282) for model wrapping
- Patches applied via [`add_wrapper_with_key()`](nodes.py:281) and [`set_model_patch()`](nodes.py:288)
- Audio embeddings stored in `model_options["transformer_options"]["audio_embeds"]`
- Two patch types: `attn2_patch` (cross-attention) and optional `attn1_patch` (self-attention for reference masks)

### Audio Processing
- Linear interpolation from 50fps to 25fps using [`linear_interpolation()`](nodes.py:20)
- Audio encoder output validated against model patch dimensions ([`encode_audio_features()`](nodes.py:162))
- Multi-speaker: masks concatenated via `torch.cat([mask_1, mask_2])`, combined with "add" mode

### Video Generation Flow
- **Base pass**: Generates initial `length` frames from motion frame (start image or zeros)
- **Extend passes**: Chains generation using last `motion_frame_count` frames as motion conditioning
- Pass count calculated from audio duration: `total_passes = 1 + ceil((total_frames - length) / (length - motion_frame_count))`
- Latent temporal dimension: `((length - 1) // 4) + 1`

### Device Handling
- Always use [`comfy.model_management.intermediate_device()`](nodes.py:495) for tensor creation
- Final samples moved to intermediate device: `samples.to(comfy.model_management.intermediate_device())`

## Two Node Variants

1. **InfiniteTalkAutoSampler** ([lines 323-621](nodes.py:323)): Standard KSampler interface with seed/steps/cfg/sampler_name/scheduler/denoise
2. **InfiniteTalkAutoSamplerAdvanced** ([lines 629-910](nodes.py:629)): Custom sampler interface accepting NOISE/SAMPLER/SIGMAS objects for advanced workflows

## Critical Implementation Details

### Motion Frame Handling
- Base pass uses start image's first frame latent if provided, else zeros ([`nodes.py:502-508`](nodes.py:502))
- Extend pass encodes last `motion_frame_count` RGB frames to latent ([`nodes.py:573-580`](nodes.py:573))
- Motion frames must be encoded separately per pass (not cached)

### Audio Feature Validation
- Audio encoder dimensions must match `model_patch.model.audio_proj.blocks` and `.channels` ([`nodes.py:178-187`](nodes.py:178))
- Compatible encoders: wav2vec2-base (blocks=12, channels=768)

### Latent Shape Convention
- Shape format: `[batch, channels, temporal, height//8, width//8]` where channels=16 for WAN
- Temporal calculated as `((length - 1) // 4) + 1` due to 4x temporal compression

## Dependencies

Runtime imports from ComfyUI core (not pip packages):
- `comfy.model_management`, `comfy.sample`, `comfy.samplers`, `comfy.utils`, `comfy.patcher_extension`
- `latent_preview`, `node_helpers`
- `comfy.ldm.wan.model_multitalk` - WAN model-specific InfiniteTalk implementations
