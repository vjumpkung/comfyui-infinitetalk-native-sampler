# CLAUDE.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

ComfyUI custom node package for InfiniteTalk - generates lip-sync videos from audio using WAN 2.1 models with automatic frame extension for long audio.

## File Structure

- [`nodes.py`](nodes.py) — all node logic and shared helpers
- [`__init__.py`](__init__.py) — node registration; also imports `_patches` at startup
- [`_patches.py`](_patches.py) — monkey-patches applied at import time to fix ComfyUI core compatibility issues (e.g. wav2vec2 fp16 dtype bug)

## Architecture Patterns

### ComfyUI Node Structure
- Nodes defined in [`nodes.py`](nodes.py) as V3 classes extending `io.ComfyNode` with `define_schema()` and `execute()` classmethods
- Imported from `comfy_api.latest`: `io`, `ComfyExtension`
- Types use native `io.*` classes: `io.Vae`, `io.Clip`, `io.ModelPatch`, `io.AudioEncoderOutput`, `io.Noise`, `io.Sampler`, `io.Sigmas`, `io.ClipVisionOutput`, `io.Audio`
- Node registration in [`__init__.py`](__init__.py) via `ComfyExtension.get_node_list()` and `comfy_entrypoint()`

### Model Patching System
- Uses [`comfy.patcher_extension.WrappersMP.OUTER_SAMPLE`](nodes.py:296) for model wrapping
- Patches applied via [`add_wrapper_with_key()`](nodes.py:296) and [`set_model_patch()`](nodes.py:303)
- Audio embeddings stored in `model_options["transformer_options"]["audio_embeds"]`
- Two patch types: `attn2_patch` (cross-attention) and optional `attn1_patch` (self-attention for reference masks)

### Audio Processing
- Linear interpolation from 50fps to `framerate` fps using [`linear_interpolation()`](nodes.py:20)
- `framerate` is a user-supplied INT input (default 25) passed through `compute_pass_counts()` and `encode_audio_features()`
- Audio encoder output validated against model patch dimensions ([`encode_audio_features()`](nodes.py:162))
- Multi-speaker: masks concatenated via `torch.cat([mask_1, mask_2])`, combined with "add" mode

### Video Generation Flow
- **Base pass**: Generates initial `length` frames from motion frame (start image or zeros)
- **Extend passes**: Chains generation using last `motion_frame_count` frames as motion conditioning
- Pass count calculated from audio duration: `total_passes = 1 + ceil((total_frames - length) / (length - motion_frame_count))`
- Latent temporal dimension: `((length - 1) // 4) + 1`
- Pixel frames accumulated per pass (not latents) — required by WAN's causal VAE

### Progress Bar
- Single `ProgressBar(total_passes * steps_per_pass)` spans the entire generation
- `make_pass_callback()` wraps `latent_preview.prepare_callback` and calls `pbar.update_absolute(step_offset + step + 1)` for per-step updates across all passes
- `common_ksampler` and `advanced_sampler` accept an optional `callback` parameter; pass methods always supply one

### Device Handling
- Always use `comfy.model_management.intermediate_device()` for tensor creation
- Final samples moved to intermediate device: `samples.to(comfy.model_management.intermediate_device())`

## Two Node Variants

1. **InfiniteTalkAutoSampler**: Standard KSampler interface with seed/steps/cfg/sampler_name/scheduler/denoise/framerate
2. **InfiniteTalkAutoSamplerAdvanced**: Custom sampler interface accepting NOISE/SAMPLER/SIGMAS objects with framerate for advanced workflows

## Critical Implementation Details

### Motion Frame Handling
- Base pass uses start image's first latent frame if provided, else zeros
- Extend pass VAE-encodes the last `motion_frame_count` RGB frames from accumulated pixel output
- Motion frames **must** be re-encoded from decoded pixels each pass — WAN uses a causal VAE where latent frame 0 is a keyframe (absolute encoding); slicing inter-frame latents from prior pass output and reusing them as position-0 motion frames causes color shift artifacts

### Audio Feature Validation
- Audio encoder dimensions must match `model_patch.model.audio_proj.blocks` and `.channels`
- Compatible encoders: wav2vec2-base (blocks=12, channels=768)

### Latent Shape Convention
- Shape format: `[batch, channels, temporal, height//8, width//8]` where channels=16 for WAN
- Temporal calculated as `((length - 1) // 4) + 1` due to 4x temporal compression

### Causal VAE Constraint
- WAN's temporal VAE is causal: frame 0 is a keyframe, subsequent frames are inter-frames relative to prior context
- You cannot arbitrarily concatenate latent tensors from separate sampling passes and decode them together — inter-frame latents from pass N depend on pass N's keyframe context
- Each pass's output must be decoded independently; pixel frames are accumulated, not latents

## Monkey-Patch System (`_patches.py`)

- Applied automatically at import via `__init__.py`
- Current patch: `PositionalConvEmbedding.forward` in `comfy.audio_encoders.wav2vec2` — casts input to conv weight dtype to fix `RuntimeError: Input type (float) and bias type (c10::Half) should be the same` when the wav2vec2 audio encoder is loaded in fp16
- Pattern: wrap original class method, apply fix, reassign; wrapped in try/except to fail gracefully

## Dependencies

Runtime imports from ComfyUI core (not pip packages):
- `comfy_api.latest` — V3 node API (`io`, `ComfyExtension`)
- `comfy.model_management`, `comfy.sample`, `comfy.samplers`, `comfy.utils`, `comfy.patcher_extension`
- `latent_preview`, `node_helpers`
- `comfy.ldm.wan.model_multitalk` - WAN model-specific InfiniteTalk implementations
