"""Bounded, generation-free GPU check of the Sage CUDA address boundary."""
import importlib
import sys
import torch


def probe():
    from sageattention.core import per_channel_fp8, per_warp_int8_cuda
    registry = importlib.import_module('nodes').NODE_CLASS_MAPPINGS
    safety = sys.modules[registry['MiniMaxH3MemoryEfficientSageAttentionPatch'].__module__]
    sequence = 205000
    with torch.inference_mode():
        # Real H3 fused-buffer stride; only one head is quantized, so this check
        # needs ~9 GB rather than loading diffusion weights or generating media.
        fused = torch.zeros((1, sequence, 3, 56, 128), device='cuda', dtype=torch.bfloat16)
        q = fused[:,:,0,:1,:]
        k = fused[:,:,1,:1,:]
        v = fused[:,:,2,:1,:]
        q.fill_(1)
        k.fill_(2)
        v.fill_(3)
        q_raw, _, _, _ = per_warp_int8_cuda(q,k,km=k.mean(dim=1,keepdim=True),tensor_layout='NHD',BLKQ=128,WARPQ=32,BLKK=64)
        v_raw, _, _ = per_channel_fp8(v,tensor_layout='NHD',smooth_v=False)
        q_safe = safety.compact_quant_input(q)
        k_safe = safety.compact_quant_input(k)
        v_safe = safety.compact_quant_input(v)
        q_fixed, _, _, _ = per_warp_int8_cuda(q_safe,k_safe,km=k_safe.mean(dim=1,keepdim=True),tensor_layout='NHD',BLKQ=128,WARPQ=32,BLKK=64)
        v_fixed, _, _ = per_channel_fp8(v_safe,tensor_layout='NHD',smooth_v=False)
        q_error = int(torch.count_nonzero(q_raw != q_fixed).item())
        v_error = int(torch.count_nonzero(v_raw.float() != v_fixed.float()).item())
        fixed_q_nonzero = bool(torch.all(q_fixed != 0).item())
        # Sage permutes each 16-position block, so padded zeroes can appear
        # before the logical length. Count actual values across the padded axis.
        fixed_v_nonzero = int(torch.count_nonzero(v_fixed.float()).item()) == v.numel()
        result = safety._sageattn_int8_fp8_nhd([q,k,v], torch.bfloat16)
        production_error = float((result.float() - 3).abs().max().item())
        production_passed = bool(torch.isfinite(result).all().item()) and production_error < 0.125
        return {'gpu':torch.cuda.get_device_name(0),'sequence':sequence,'shape':list(v.shape),'stride':list(v.stride()),
                'max_relative_offset':sum((n-1)*s for n,s in zip(v.shape,v.stride())),
                'unsafe_q_mismatched_elements':q_error,'unsafe_v_mismatched_elements':v_error,
                'safe_q_all_nonzero':fixed_q_nonzero,'safe_v_all_nonzero':fixed_v_nonzero,
                'production_max_absolute_error':production_error,
                'production_passed':production_passed,
                'passed':fixed_q_nonzero and fixed_v_nonzero and production_passed}
