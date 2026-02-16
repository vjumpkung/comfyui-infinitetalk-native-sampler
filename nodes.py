import logging
import math

import comfy.model_management
import comfy.patcher_extension
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import node_helpers
import torch
from comfy.ldm.wan.model_multitalk import (
    InfiniteTalkOuterSampleWrapper,
    MultiTalkCrossAttnPatch,
    MultiTalkGetAttnMapPatch,
    project_audio_features,
)


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = torch.nn.functional.interpolate(
        features, size=output_len, align_corners=True, mode="linear"
    )
    return output_features.transpose(1, 2)


def common_ksampler(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise=1.0,
):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_image, latent.get("downscale_ratio_spacial", None)
    )

    batch_inds = latent.get("batch_index", None)
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return out


def advanced_sampler(
    model_patched, positive, negative, cfg, noise_obj, sampler_obj, sigmas, latent
):
    """Sample using advanced custom sampler inputs (NOISE, SAMPLER, SIGMAS)."""
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model_patched, latent_image, latent.get("downscale_ratio_spacial", None)
    )

    noise_mask = latent.get("noise_mask", None)

    # Build a CFGGuider with the patched model
    guider = comfy.samplers.CFGGuider(model_patched)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)

    noise = noise_obj.generate_noise({"samples": latent_image})

    callback = latent_preview.prepare_callback(model_patched, sigmas.shape[-1] - 1)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = guider.sample(
        noise,
        latent_image,
        sampler_obj,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=noise_obj.seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return out


# --- Shared helpers ---


def validate_two_speaker(audio_encoder_output_2, mask_1, mask_2):
    if audio_encoder_output_2 is not None:
        if mask_1 is None or mask_2 is None:
            raise ValueError(
                "Both mask_1 and mask_2 are required when using two audio encoder outputs."
            )
    if mask_1 is not None or mask_2 is not None:
        if audio_encoder_output_2 is None:
            raise ValueError(
                "audio_encoder_output_2 is required when masks are provided."
            )
        if mask_1 is None or mask_2 is None:
            raise ValueError("Both mask_1 and mask_2 must be provided together.")


def compute_pass_counts(audio, length, motion_frame_count):
    fps = 25
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    audio_duration = waveform.shape[-1] / sample_rate
    total_frames = math.ceil(audio_duration * fps)
    logging.info(
        f"InfiniteTalkAutoSampler: audio={audio_duration:.2f}s, total_frames={total_frames}"
    )

    if total_frames < 1:
        raise ValueError("Audio is too short to produce any frames.")

    extend_frames = length - motion_frame_count
    if extend_frames <= 0:
        raise ValueError("length must be greater than motion_frame_count.")
    num_extends = max(0, math.ceil((total_frames - length) / extend_frames))
    total_passes = 1 + num_extends
    logging.info(
        f"InfiniteTalkAutoSampler: {total_passes} passes (1 base + {num_extends} extend)"
    )
    return total_frames, num_extends, total_passes


def encode_audio_features(audio_encoder_output_1, audio_encoder_output_2, model_patch):
    encoded_audio_list = []
    seq_lengths = []
    for audio_enc in [audio_encoder_output_1, audio_encoder_output_2]:
        if audio_enc is None:
            continue
        all_layers = audio_enc["encoded_audio_all_layers"]
        encoded_audio = torch.stack(all_layers, dim=0).squeeze(1)[1:]
        encoded_audio = linear_interpolation(
            encoded_audio, input_fps=50, output_fps=25
        ).movedim(0, 1)
        encoded_audio_list.append(encoded_audio)
        seq_lengths.append(encoded_audio.shape[0])

    # Validate audio encoder dimensions match model_patch expectations
    if encoded_audio_list:
        audio_proj = model_patch.model.audio_proj
        enc_layers = encoded_audio_list[0].shape[1]
        enc_channels = encoded_audio_list[0].shape[2]
        if enc_layers != audio_proj.blocks or enc_channels != audio_proj.channels:
            raise ValueError(
                f"Audio encoder output dimensions ({enc_layers} layers, {enc_channels} channels) "
                f"do not match InfiniteTalk model patch expectations ({audio_proj.blocks} layers, "
                f"{audio_proj.channels} channels). "
                f"Please use a compatible audio encoder model (e.g. wav2vec2-base for blocks=12/channels=768)."
            )

    # Combine multi-speaker audio with "add" mode
    if len(encoded_audio_list) > 1:
        total_len = sum(seq_lengths)
        full_list = []
        offset = 0
        for emb, seq_len in zip(encoded_audio_list, seq_lengths):
            full = torch.zeros(total_len, *emb.shape[1:], dtype=emb.dtype)
            full[offset : offset + seq_len] = emb
            full_list.append(full)
            offset += seq_len
        encoded_audio_list = full_list

    return encoded_audio_list


def encode_start_image(start_image, vae, width, height, length):
    if start_image is None:
        return None
    latent_temporal = ((length - 1) // 4) + 1
    si = comfy.utils.common_upscale(
        start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
    image = (
        torch.ones(
            (length, height, width, si.shape[-1]), device=si.device, dtype=si.dtype
        )
        * 0.5
    )
    image[: si.shape[0]] = si

    concat_latent_image = vae.encode(image[:, :, :, :3])
    concat_mask = torch.ones(
        (
            1,
            1,
            latent_temporal,
            concat_latent_image.shape[-2],
            concat_latent_image.shape[-1],
        ),
        device=si.device,
        dtype=si.dtype,
    )
    concat_mask[:, :, : ((si.shape[0] - 1) // 4) + 1] = 0.0
    return {
        "concat_latent_image": concat_latent_image,
        "concat_mask": concat_mask,
    }


def prepare_ref_masks(ref_masks, latent):
    if ref_masks is None:
        return None
    token_ref_target_masks = torch.nn.functional.interpolate(
        ref_masks.unsqueeze(0),
        size=(latent.shape[-2] // 2, latent.shape[-1] // 2),
        mode="nearest",
    )[0]
    token_ref_target_masks = (token_ref_target_masks > 0).view(
        token_ref_target_masks.shape[0], -1
    )
    return token_ref_target_masks


def decode_video(vae, latent_samples):
    frames = vae.decode(latent_samples)
    if len(frames.shape) == 5:
        frames = frames.reshape(
            -1, frames.shape[-3], frames.shape[-2], frames.shape[-1]
        )
    return frames


def patch_model_for_pass(
    model,
    model_patch,
    audio_scale,
    encoded_audio_list,
    ref_masks,
    latent,
    motion_frames_latent,
    audio_start,
    audio_end,
    is_extend,
):
    """Clone model and apply all InfiniteTalk patches for one sampling pass."""
    model_patched = model.clone()

    audio_embed = project_audio_features(
        model_patch.model.audio_proj, encoded_audio_list, audio_start, audio_end
    ).to(model_patched.model_dtype())
    model_patched.model_options["transformer_options"]["audio_embeds"] = audio_embed

    model_patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        "infinite_talk_outer_sample",
        InfiniteTalkOuterSampleWrapper(
            motion_frames_latent, model_patch, is_extend=is_extend
        ),
    )
    model_patched.set_model_patch(
        MultiTalkCrossAttnPatch(model_patch, audio_scale), "attn2_patch"
    )

    token_ref_target_masks = prepare_ref_masks(ref_masks, latent)
    if token_ref_target_masks is not None:
        model_patched.set_model_patch(
            MultiTalkGetAttnMapPatch(token_ref_target_masks), "attn1_patch"
        )

    return model_patched


def prepare_conditioning(positive, negative, start_image_cond, clip_vision_output):
    """Apply start_image and clip_vision conditioning."""
    pos = positive
    neg = negative
    if start_image_cond is not None:
        pos = node_helpers.conditioning_set_values(pos, start_image_cond)
        neg = node_helpers.conditioning_set_values(neg, start_image_cond)
    if clip_vision_output is not None:
        pos = node_helpers.conditioning_set_values(
            pos, {"clip_vision_output": clip_vision_output}
        )
        neg = node_helpers.conditioning_set_values(
            neg, {"clip_vision_output": clip_vision_output}
        )
    return pos, neg


# ===========================================================================
# InfiniteTalkAutoSampler — basic KSampler interface
# ===========================================================================


class InfiniteTalkAutoSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "model_patch": ("MODEL_PATCH",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "audio_encoder_output_1": ("AUDIO_ENCODER_OUTPUT",),
                "audio": ("AUDIO",),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                "motion_frame_count": (
                    "INT",
                    {"default": 9, "min": 1, "max": 33, "step": 1},
                ),
                "audio_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "audio_encoder_output_2": ("AUDIO_ENCODER_OUTPUT",),
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "execute"
    CATEGORY = "video/infinitetalk"

    def execute(
        self,
        model,
        model_patch,
        positive,
        negative,
        vae,
        audio_encoder_output_1,
        audio,
        width,
        height,
        length,
        motion_frame_count,
        audio_scale,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        start_image=None,
        clip_vision_output=None,
        audio_encoder_output_2=None,
        mask_1=None,
        mask_2=None,
    ):
        validate_two_speaker(audio_encoder_output_2, mask_1, mask_2)
        total_frames, num_extends, total_passes = compute_pass_counts(
            audio, length, motion_frame_count
        )

        ref_masks = None
        if mask_1 is not None and mask_2 is not None:
            ref_masks = torch.cat([mask_1, mask_2])

        encoded_audio_list = encode_audio_features(
            audio_encoder_output_1, audio_encoder_output_2, model_patch
        )
        start_image_cond = encode_start_image(start_image, vae, width, height, length)

        pbar = comfy.utils.ProgressBar(total_passes)

        # --- Base pass ---
        accumulated_frames = self._run_base_pass(
            model,
            model_patch,
            positive,
            negative,
            vae,
            width,
            height,
            length,
            audio_scale,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            start_image_cond,
            clip_vision_output,
            encoded_audio_list,
            ref_masks,
        )
        pbar.update(1)

        # --- Extend passes ---
        for ext_idx in range(num_extends):
            accumulated_frames = self._run_extend_pass(
                model,
                model_patch,
                positive,
                negative,
                vae,
                width,
                height,
                length,
                motion_frame_count,
                audio_scale,
                seed + ext_idx + 1,
                steps,
                cfg,
                sampler_name,
                scheduler,
                denoise,
                start_image_cond,
                clip_vision_output,
                encoded_audio_list,
                ref_masks,
                accumulated_frames,
            )
            pbar.update(1)

        if accumulated_frames.shape[0] > total_frames:
            accumulated_frames = accumulated_frames[:total_frames]

        return (accumulated_frames, audio)

    def _run_base_pass(
        self,
        model,
        model_patch,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        audio_scale,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        start_image_cond,
        clip_vision_output,
        encoded_audio_list,
        ref_masks,
    ):
        latent = torch.zeros(
            [1, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )

        pos, neg = prepare_conditioning(
            positive, negative, start_image_cond, clip_vision_output
        )

        if start_image_cond is not None:
            motion_frames_latent = start_image_cond["concat_latent_image"][:, :, :1]
        else:
            motion_frames_latent = torch.zeros(
                [1, 16, 1, height // 8, width // 8],
                device=comfy.model_management.intermediate_device(),
            )

        model_patched = patch_model_for_pass(
            model,
            model_patch,
            audio_scale,
            encoded_audio_list,
            ref_masks,
            latent,
            motion_frames_latent,
            audio_start=0,
            audio_end=length,
            is_extend=False,
        )

        out_latent = common_ksampler(
            model_patched,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            pos,
            neg,
            {"samples": latent},
            denoise,
        )

        frames = decode_video(vae, out_latent["samples"])
        logging.info(f"InfiniteTalk base pass: decoded {frames.shape[0]} frames")
        return frames

    def _run_extend_pass(
        self,
        model,
        model_patch,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        motion_frame_count,
        audio_scale,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        start_image_cond,
        clip_vision_output,
        encoded_audio_list,
        ref_masks,
        accumulated_frames,
    ):
        latent = torch.zeros(
            [1, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )

        pos, neg = prepare_conditioning(
            positive, negative, start_image_cond, clip_vision_output
        )

        motion_frames = comfy.utils.common_upscale(
            accumulated_frames[-motion_frame_count:].movedim(-1, 1),
            width,
            height,
            "bilinear",
            "center",
        ).movedim(1, -1)
        motion_frames_latent = vae.encode(motion_frames[:, :, :, :3])

        frame_offset = accumulated_frames.shape[0] - motion_frame_count
        audio_start = frame_offset
        audio_end = audio_start + length
        logging.info(
            f"InfiniteTalk extend pass: audio frames {audio_start} - {audio_end}"
        )

        model_patched = patch_model_for_pass(
            model,
            model_patch,
            audio_scale,
            encoded_audio_list,
            ref_masks,
            latent,
            motion_frames_latent,
            audio_start,
            audio_end,
            is_extend=True,
        )

        out_latent = common_ksampler(
            model_patched,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            pos,
            neg,
            {"samples": latent},
            denoise,
        )

        frames = decode_video(vae, out_latent["samples"])
        new_frames = frames[motion_frame_count:]
        accumulated_frames = torch.cat([accumulated_frames, new_frames], dim=0)
        logging.info(
            f"InfiniteTalk extend pass: total accumulated {accumulated_frames.shape[0]} frames"
        )
        return accumulated_frames


# ===========================================================================
# InfiniteTalkAutoSamplerAdvanced — custom sampler interface (NOISE/SAMPLER/SIGMAS)
# ===========================================================================


class InfiniteTalkAutoSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "model_patch": ("MODEL_PATCH",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "audio_encoder_output_1": ("AUDIO_ENCODER_OUTPUT",),
                "audio": ("AUDIO",),
                "noise": ("NOISE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                "motion_frame_count": (
                    "INT",
                    {"default": 9, "min": 1, "max": 33, "step": 1},
                ),
                "audio_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "audio_encoder_output_2": ("AUDIO_ENCODER_OUTPUT",),
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "execute"
    CATEGORY = "video/infinitetalk"

    def execute(
        self,
        model,
        model_patch,
        positive,
        negative,
        vae,
        audio_encoder_output_1,
        audio,
        noise,
        sampler,
        sigmas,
        width,
        height,
        length,
        motion_frame_count,
        audio_scale,
        cfg,
        start_image=None,
        clip_vision_output=None,
        audio_encoder_output_2=None,
        mask_1=None,
        mask_2=None,
    ):
        validate_two_speaker(audio_encoder_output_2, mask_1, mask_2)
        total_frames, num_extends, total_passes = compute_pass_counts(
            audio, length, motion_frame_count
        )

        ref_masks = None
        if mask_1 is not None and mask_2 is not None:
            ref_masks = torch.cat([mask_1, mask_2])

        encoded_audio_list = encode_audio_features(
            audio_encoder_output_1, audio_encoder_output_2, model_patch
        )
        start_image_cond = encode_start_image(start_image, vae, width, height, length)

        pbar = comfy.utils.ProgressBar(total_passes)

        # --- Base pass ---
        accumulated_frames = self._run_base_pass(
            model,
            model_patch,
            positive,
            negative,
            vae,
            width,
            height,
            length,
            audio_scale,
            cfg,
            noise,
            sampler,
            sigmas,
            start_image_cond,
            clip_vision_output,
            encoded_audio_list,
            ref_masks,
        )
        pbar.update(1)

        # --- Extend passes ---
        for ext_idx in range(num_extends):
            accumulated_frames = self._run_extend_pass(
                model,
                model_patch,
                positive,
                negative,
                vae,
                width,
                height,
                length,
                motion_frame_count,
                audio_scale,
                cfg,
                noise,
                sampler,
                sigmas,
                start_image_cond,
                clip_vision_output,
                encoded_audio_list,
                ref_masks,
                accumulated_frames,
            )
            pbar.update(1)

        if accumulated_frames.shape[0] > total_frames:
            accumulated_frames = accumulated_frames[:total_frames]

        return (accumulated_frames, audio)

    def _run_base_pass(
        self,
        model,
        model_patch,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        audio_scale,
        cfg,
        noise_obj,
        sampler_obj,
        sigmas,
        start_image_cond,
        clip_vision_output,
        encoded_audio_list,
        ref_masks,
    ):
        latent = torch.zeros(
            [1, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )

        pos, neg = prepare_conditioning(
            positive, negative, start_image_cond, clip_vision_output
        )

        if start_image_cond is not None:
            motion_frames_latent = start_image_cond["concat_latent_image"][:, :, :1]
        else:
            motion_frames_latent = torch.zeros(
                [1, 16, 1, height // 8, width // 8],
                device=comfy.model_management.intermediate_device(),
            )

        model_patched = patch_model_for_pass(
            model,
            model_patch,
            audio_scale,
            encoded_audio_list,
            ref_masks,
            latent,
            motion_frames_latent,
            audio_start=0,
            audio_end=length,
            is_extend=False,
        )

        out_latent = advanced_sampler(
            model_patched,
            pos,
            neg,
            cfg,
            noise_obj,
            sampler_obj,
            sigmas,
            {"samples": latent},
        )

        frames = decode_video(vae, out_latent["samples"])
        logging.info(
            f"InfiniteTalk base pass (advanced): decoded {frames.shape[0]} frames"
        )
        return frames

    def _run_extend_pass(
        self,
        model,
        model_patch,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        motion_frame_count,
        audio_scale,
        cfg,
        noise_obj,
        sampler_obj,
        sigmas,
        start_image_cond,
        clip_vision_output,
        encoded_audio_list,
        ref_masks,
        accumulated_frames,
    ):
        latent = torch.zeros(
            [1, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )

        pos, neg = prepare_conditioning(
            positive, negative, start_image_cond, clip_vision_output
        )

        motion_frames = comfy.utils.common_upscale(
            accumulated_frames[-motion_frame_count:].movedim(-1, 1),
            width,
            height,
            "bilinear",
            "center",
        ).movedim(1, -1)
        motion_frames_latent = vae.encode(motion_frames[:, :, :, :3])

        frame_offset = accumulated_frames.shape[0] - motion_frame_count
        audio_start = frame_offset
        audio_end = audio_start + length
        logging.info(
            f"InfiniteTalk extend pass (advanced): audio frames {audio_start} - {audio_end}"
        )

        model_patched = patch_model_for_pass(
            model,
            model_patch,
            audio_scale,
            encoded_audio_list,
            ref_masks,
            latent,
            motion_frames_latent,
            audio_start,
            audio_end,
            is_extend=True,
        )

        out_latent = advanced_sampler(
            model_patched,
            pos,
            neg,
            cfg,
            noise_obj,
            sampler_obj,
            sigmas,
            {"samples": latent},
        )

        frames = decode_video(vae, out_latent["samples"])
        new_frames = frames[motion_frame_count:]
        accumulated_frames = torch.cat([accumulated_frames, new_frames], dim=0)
        logging.info(
            f"InfiniteTalk extend pass (advanced): total accumulated {accumulated_frames.shape[0]} frames"
        )
        return accumulated_frames
