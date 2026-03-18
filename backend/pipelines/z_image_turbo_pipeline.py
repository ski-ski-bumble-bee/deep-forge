"""
Z-Image Turbo training pipeline.

Architecture recap:
  - S3-DiT: 30 transformer layers, hidden=3840, 32 heads, FFN=10240, 6.15B params
  - Text encoder: Qwen3-4B (frozen)
  - VAE: Flux VAE (ae.safetensors, frozen)
  - Single-stream: text tokens + image latent tokens concatenated
  - Flow matching: linear interpolation x_t = (1-t)*x_0 + t*noise
  - Model predicts velocity v = dx/dt
  - Turbo: distilled for 8 NFE, guidance_scale=0.0

For LoRA character training on the distilled model:
  - Target specific attention blocks (not all 30 — avoids breaking distillation)
  - Recommended: target middle blocks (e.g. blocks 8-22) attention layers
  - Keep lr low (1e-5 to 5e-5) to preserve distillation quality
  - Use flow matching loss

Key LoRA target patterns for identity preservation:
  - 'transformer_blocks\\.([8-9]|1[0-9]|2[0-2])\\.attention' — middle attention blocks
  - 'attention\\.to_q|attention\\.to_k|attention\\.to_v|attention\\.to_out' — QKV + output projections
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from backend.pipelines.base_pipeline import (
    BaseDiffusionPipeline,
    PipelineComponents,
    SampleRequest,
    SampleResult,
)
from backend.pipelines.registry import register_pipeline
from backend.pipelines.samplers import create_sampler


class ZImageTurboPipeline(BaseDiffusionPipeline):
    """
    Training + inference pipeline for Z-Image Turbo (S3-DiT).

    Components expected in ComponentBundle:
      - 'dit' or 'denoiser': the S3-DiT transformer (6.15B)
      - 'text_encoder': Qwen3-4B
      - 'vae': Flux VAE (ae.safetensors)

    Flow matching formulation:
      x_t = (1 - t) * x_0 + t * eps     (linear interpolation)
      target = eps - x_0                   (velocity = dx/dt)
      loss = ||model(x_t, t, cond) - target||^2
    """

    name = "zimage_turbo"
    display_name = "Z-Image Turbo"
    description = "S3-DiT flow matching pipeline for Z-Image Turbo (6.15B)"
    uses_flow_matching = True
    default_num_steps = 8
    default_guidance_scale = 0.0
    default_sampler = "euler"

    # VAE scale factor (Flux VAE uses 16-channel latents with 8x downscale)
    vae_scale_factor = 8
    vae_latent_channels = 16

    def __init__(self, components: PipelineComponents, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16,
                 # Flow matching params
                 shift: float = 1.0,
                 # Loss weighting
                 loss_type: str = "mse",
                 snr_gamma: Optional[float] = None,
                 ):
        super().__init__(components, device, dtype)
        self.shift = shift
        self.loss_type = loss_type
        self.snr_gamma = snr_gamma

        self.training_adapter = components.extras.get('training_adapter', None)

    # ── Text encoding ──

    def encode_prompt(self, prompt: str, negative_prompt: Optional[str] = None
                      ) -> Dict[str, torch.Tensor]:
        """
        Encode text with Qwen3-4B.

        The text encoder produces hidden states that are concatenated
        with image latent tokens in the S3-DiT's single stream.
        """
        te = self.components.text_encoder
        if te is None:
            raise RuntimeError("Text encoder not loaded")

        # The actual encoding depends on how Qwen3 is wrapped.
        # In diffusers, ZImagePipeline handles tokenization internally.
        # Here we support both raw model forward and a wrapped encode() method.
        with torch.no_grad():
            if hasattr(te, 'encode'):
                # Wrapped encoder with tokenizer built in
                embeds = te.encode(prompt)
            elif hasattr(te, 'tokenizer'):
                tokens = te.tokenizer(
                    prompt, return_tensors='pt', padding=True,
                    truncation=True, max_length=512
                ).to(self.device)
                embeds = te(**tokens).last_hidden_state
            else:
                # Assume pre-tokenized or the model handles strings
                embeds = te(prompt)

        result = {'prompt_embeds': embeds}

        if negative_prompt is not None:
            with torch.no_grad():
                if hasattr(te, 'encode'):
                    neg_embeds = te.encode(negative_prompt)
                elif hasattr(te, 'tokenizer'):
                    neg_tokens = te.tokenizer(
                        negative_prompt, return_tensors='pt', padding=True,
                        truncation=True, max_length=512
                    ).to(self.device)
                    neg_embeds = te(**neg_tokens).last_hidden_state
                else:
                    neg_embeds = te(negative_prompt)
                result['negative_prompt_embeds'] = neg_embeds

        return result

    # ── VAE ──

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space image to latent space via Flux VAE."""
        vae = self.components.vae
        if vae is None:
            raise RuntimeError("VAE not loaded")

        with torch.no_grad():
            if hasattr(vae, 'encode'):
                # Standard diffusers-style VAE
                enc = vae.encode(image.to(self.dtype))
                if hasattr(enc, 'latent_dist'):
                    latents = enc.latent_dist.sample()
                elif hasattr(enc, 'latents'):
                    latents = enc.latents
                else:
                    latents = enc
            else:
                latents = vae(image.to(self.dtype))

        # Scale if VAE has a scaling factor
        if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
            latents = latents * vae.config.scaling_factor

        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel space."""
        vae = self.components.vae
        if vae is None:
            raise RuntimeError("VAE not loaded")

        # Unscale
        if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
            latents = latents / vae.config.scaling_factor

        with torch.no_grad():
            if hasattr(vae, 'decode'):
                dec = vae.decode(latents.to(self.dtype))
                if hasattr(dec, 'sample'):
                    images = dec.sample
                else:
                    images = dec
            else:
                images = vae(latents.to(self.dtype))

        # Clamp to [0, 1]
        images = (images / 2 + 0.5).clamp(0, 1)
        return images

    # ── Flow matching noise schedule ──

    def get_noise(self, latent_shape: Tuple[int, ...],
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample Gaussian noise."""
        return torch.randn(latent_shape, device=self.device, dtype=self.dtype,
                           generator=generator)

    def get_timesteps(self, batch_size: int,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Sample random timesteps in [0, 1] for flow matching.

        For distilled models like Turbo, we use uniform sampling.
        The shift parameter can be used for logit-normal or shifted distributions.
        """
        # Uniform sampling in [0, 1]
        t = torch.rand(batch_size, device=self.device, dtype=self.dtype,
                        generator=generator)

        # Apply shift if needed (shifts distribution towards higher noise levels)
        if self.shift != 1.0:
            t = self.shift * t / (1 + (self.shift - 1) * t)

        return t

    def add_noise(self, clean_latents: torch.Tensor, noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """
        Flow matching forward process:
          x_t = (1 - t) * x_0 + t * eps

        timesteps shape: [B] → reshape for broadcasting
        """
        t = timesteps
        while t.dim() < clean_latents.dim():
            t = t.unsqueeze(-1)

        noisy = (1.0 - t) * clean_latents + t * noise
        return noisy

    def compute_target(self, clean_latents: torch.Tensor, noise: torch.Tensor,
                       timesteps: torch.Tensor) -> torch.Tensor:
        """
        Flow matching target: velocity = eps - x_0 = dx/dt
        (derivative of x_t = (1-t)*x_0 + t*eps w.r.t. t)
        """
        return noise - clean_latents

    # ── Denoiser forward ──

    def forward_denoise(self, noisy_latents: torch.Tensor,
                        timesteps: torch.Tensor,
                        condition: Dict[str, torch.Tensor],
                        **kwargs) -> torch.Tensor:
        """
        Run the S3-DiT denoiser.

        The S3-DiT takes:
          - hidden_states: noisy latent tokens (after patchification)
          - timestep: scalar or [B] timesteps
          - encoder_hidden_states: text embeddings from Qwen3
        """
        dit = self.components.denoiser
        if dit is None:
            raise RuntimeError("Denoiser (DiT) not loaded")

        prompt_embeds = condition.get('prompt_embeds')

        # The actual S3-DiT forward signature varies by implementation.
        # Support both diffusers-style and raw model forward.
        if hasattr(dit, 'forward'):
            # Try diffusers-style ZImageTransformer2DModel
            try:
                out = dit(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )
                if isinstance(out, tuple):
                    return out[0]
                return out
            except TypeError:
                # Fallback: maybe it takes positional args
                out = dit(noisy_latents, timesteps, prompt_embeds)
                if isinstance(out, tuple):
                    return out[0]
                return out

        raise RuntimeError("Denoiser has no compatible forward method")

    # ── Loss ──

    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor,
                     timesteps: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute flow matching loss.

        For distilled models, MSE is standard.
        Optional: min-SNR weighting to stabilize training.
        """
        if self.loss_type == "huber":
            loss = F.huber_loss(prediction, target, reduction='none')
        else:
            loss = F.mse_loss(prediction, target, reduction='none')

        # Per-sample loss: mean over spatial dims
        loss = loss.mean(dim=list(range(1, loss.dim())))

        # Optional SNR weighting
        if self.snr_gamma is not None:
            # For flow matching, SNR ≈ (1-t)²/t²
            t = timesteps.clamp(1e-5, 1 - 1e-5)
            snr = ((1 - t) / t) ** 2
            weight = torch.clamp(snr, max=self.snr_gamma) / snr
            loss = loss * weight

        return loss.mean()

    # ── Inference / Sampling ──

    @torch.no_grad()
    def sample(self, request: SampleRequest) -> SampleResult:
        """
        Full inference loop for generating preview samples during training.

        For Z-Image Turbo:
          - 8 steps (NFE)
          - guidance_scale = 0.0 (no CFG, it's distilled)
          - Euler sampler
        """
        # Use defaults if not specified
        num_steps = request.num_steps or self.default_num_steps
        guidance = request.guidance_scale if request.guidance_scale is not None else self.default_guidance_scale
        sampler_name = request.sampler or self.default_sampler

        sampler = create_sampler(sampler_name, num_steps=num_steps, shift=self.shift)
        schedule = sampler.get_schedule(device=self.device)

        all_images = []
        all_seeds = []

        for i, prompt in enumerate(request.prompts):
            seed = (request.seed + i) if request.seed is not None else torch.randint(0, 2**32, (1,)).item()
            gen = torch.Generator(device=self.device).manual_seed(seed)

            # Encode prompt
            cond = self.encode_prompt(prompt, request.negative_prompts[i]
                                      if request.negative_prompts and i < len(request.negative_prompts)
                                      else None)

            # Initial noise
            latent_shape = self.get_latent_shape(request.height, request.width, batch_size=1)
            latents = self.get_noise(latent_shape, generator=gen)

            # Denoise loop
            if hasattr(sampler, 'reset'):
                sampler.reset()

            for step_idx in range(num_steps):
                t_current = schedule[step_idx]
                t_next = schedule[step_idx + 1]

                # Model prediction
                t_tensor = torch.tensor([t_current], device=self.device, dtype=self.dtype)
                pred = self.forward_denoise(latents, t_tensor, cond)

                # Sampler step
                latents = sampler.step(pred, t_current.item(), t_next.item(), latents,
                                       generator=gen) if hasattr(sampler.step, '__code__') and \
                    sampler.step.__code__.co_varnames[:6].__contains__('generator') else \
                    sampler.step(pred, t_current.item(), t_next.item(), latents)

            # Decode
            pixels = self.decode_latents(latents)

            # Convert to PIL
            img = pixels[0].cpu().float().permute(1, 2, 0).numpy()
            img = (img * 255).clip(0, 255).astype('uint8')
            pil_img = Image.fromarray(img)

            all_images.append(pil_img)
            all_seeds.append(seed)

        return SampleResult(
            images=all_images,
            seeds=all_seeds,
            prompts=request.prompts,
            step=0,  # Will be set by caller
            epoch=0,
        )

    # ── Latent shape ──

    def get_latent_shape(self, height: int, width: int, batch_size: int = 1
                         ) -> Tuple[int, ...]:
        """Flux VAE: 16 channels, 8x downscale."""
        return (
            batch_size,
            self.vae_latent_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )


# ── Recommended LoRA configs for Z-Image Turbo ──

ZIMAGE_TURBO_LORA_PRESETS = {
    "character_identity": {
        "description": "Target middle attention blocks for identity preservation. "
                       "Keeps distillation intact while learning face/body features.",
        "target_patterns": [
            # Middle blocks attention (blocks 8-22 out of 0-29)
            r"transformer_blocks\.([89]|1[0-9]|2[0-2])\.attention\.(to_q|to_k|to_v|to_out)",
        ],
        "rank": 16,
        "alpha": 16,
        "recommended_lr": 5e-5,
        "recommended_steps": 3000,
    },
    "character_identity_light": {
        "description": "Lighter version — fewer blocks, smaller rank. "
                       "Good for 5-15 reference images.",
        "target_patterns": [
            r"transformer_blocks\.(1[0-8])\.attention\.(to_q|to_v)",
        ],
        "rank": 8,
        "alpha": 8,
        "recommended_lr": 3e-5,
        "recommended_steps": 2000,
    },
    "style_transfer": {
        "description": "Target FFN layers in addition to attention for style learning.",
        "target_patterns": [
            r"transformer_blocks\.(5|[6-9]|1[0-9]|2[0-5])\.attention\.(to_q|to_k|to_v)",
            r"transformer_blocks\.(1[0-9]|2[0-4])\.ff\.net",
        ],
        "rank": 32,
        "alpha": 32,
        "recommended_lr": 1e-4,
        "recommended_steps": 4000,
    },
    "full_all_layers": {
        "description": "Target all attention layers. Higher capacity but risks "
                       "breaking distillation. Use with caution and low LR.",
        "target_patterns": [
            r"attention\.(to_q|to_k|to_v|to_out)",
        ],
        "rank": 64,
        "alpha": 64,
        "recommended_lr": 1e-5,
        "recommended_steps": 4000,
    },
}


# Register
register_pipeline("zimage_turbo", ZImageTurboPipeline)
