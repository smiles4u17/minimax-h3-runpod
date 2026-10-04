import torch
import torch.nn.functional as F
from .reference_geometry import reference_size


class H3ReferenceVideoScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'width': ('INT', {'min':32,'default':832}), 'height': ('INT', {'min':32,'default':480})}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'scale'
    CATEGORY = 'video/minimax'

    def scale(self, images, width, height):
        h, w = images.shape[1:3]
        tw, th = reference_size(w, h, width * height)
        if (w, h) == (tw, th):
            return (images,)
        parts = []
        # Bound the resize workspace instead of expanding all video frames at once.
        for batch in images.split(8):
            parts.append(F.interpolate(batch.movedim(-1,1), size=(th,tw), mode='bilinear', align_corners=False, antialias=True).movedim(1,-1))
        print(f'H3_REFERENCE_RESIZE {w}x{h} -> {tw}x{th}; frames={len(images)}')
        return (torch.cat(parts),)
