import nodes


class H3ReferenceVideoScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'width': ('INT', {'min':32,'default':832}), 'height': ('INT', {'min':32,'default':480})}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'scale'
    CATEGORY = 'video/minimax'

    def scale(self, images, width, height):
        # Compatibility for saved custom graphs using the previous helper.
        # Use the same pinned KJ node/settings as newly assembled preset graphs.
        h, w = images.shape[1:3]
        result = nodes.NODE_CLASS_MAPPINGS['ImageResizeKJv2']().resize(
            images, width, height, keep_proportion='crop', upscale_method='nearest-exact',
            divisible_by=2, pad_color='0,0,0', crop_position='center', unique_id=None, device='cpu')
        print(f'H3_REFERENCE_RESIZE {w}x{h} -> {result[1]}x{result[2]}; frames={len(images)}; nearest-exact center crop cpu')
        return (result[0],)
