import math


def reference_size(width, height, area):
    scale = min(1.0, math.sqrt(area / (width * height)))
    # Downscale only and preserve the reference aspect ratio; no frame trimming.
    return max(32, int(width * scale / 32) * 32), max(32, int(height * scale / 32) * 32)
