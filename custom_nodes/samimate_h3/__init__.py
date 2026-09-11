"""SAMimate source-target masking. Requires native H3 per-token noise masks."""
import inspect
import torch
import torch.nn.functional as F
from comfy.nested_tensor import NestedTensor


def causal_mask(mask, latent_t, height, width):
    """Conservative 1,4,4,4,4 temporal unions and 32px token coverage."""
    expected = sum((1, 4, 4, 4, 4)[i % 5] for i in range(latent_t))
    if len(mask) != expected:
        raise ValueError(f"Tracked mask needs {expected} frames, received {len(mask)}")
    pooled = F.adaptive_max_pool2d((mask > 0.5).float()[:, None], (height // 2, width // 2))
    groups, start = [], 0
    for i in range(latent_t):
        end = start + (1, 4, 4, 4, 4)[i % 5]
        groups.append(pooled[start:end].amax(dim=0))
        start = end
    return torch.stack(groups, dim=1)[None].repeat_interleave(2, -2).repeat_interleave(2, -1)


class SAMimateH3MaskedTarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"target": ("LATENT",), "source": ("IMAGE",),
                             "mask_frames": ("IMAGE",), "vae": ("VAE",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "SAMimate/H3"

    def encode(self, target, source, mask_frames, vae):
        from comfy.ldm.minimax.model import MiniMaxH3Model
        if "denoise_mask" not in inspect.signature(MiniMaxH3Model.forward).parameters:
            raise RuntimeError("SAMimate requires the worker build with native H3 noise-mask support")
        video, audio = target["samples"].tensors
        length = sum((1, 4, 4, 4, 4)[i % 5] for i in range(video.shape[2]))
        if len(source) != len(mask_frames) or not 0 < len(source) <= length:
            raise ValueError("Source and tracked mask must have identical frame counts within the H3 target")
        # Only the trailing H3 grid padding repeats; never stretch a tracked mask.
        source = torch.cat((source, source[-1:].repeat(length - len(source), 1, 1, 1)))
        mask_frames = torch.cat((mask_frames, mask_frames[-1:].repeat(length - len(mask_frames), 1, 1, 1)))
        height, width = video.shape[-2:]
        pixels = F.interpolate(source[..., :3].movedim(-1, 1), (height * 16, width * 16), mode="bilinear", align_corners=False).movedim(1, -1)
        encoded = vae.encode(pixels)
        if encoded.shape != video.shape:
            raise ValueError(f"Source VAE shape {encoded.shape} differs from H3 target {video.shape}")
        mask = causal_mask(mask_frames[..., :3].mean(-1), video.shape[2], height, width).to(encoded)
        # Audio is generated internally but discarded: the console muxes original audio.
        return ({**target, "samples": NestedTensor((encoded, audio)),
                 "noise_mask": NestedTensor((mask, torch.ones_like(audio[:, :1])))},)


NODE_CLASS_MAPPINGS = {"SAMimateH3MaskedTarget": SAMimateH3MaskedTarget}
