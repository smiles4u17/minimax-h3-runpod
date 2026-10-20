"""Select an ABI-matched Sage build before either Python worker process starts.

Keep the proven Blackwell wheel in site-packages. Ada uses its own SM89-only
build so enabling 4090s cannot replace the validated Blackwell kernels.
"""
from pathlib import Path

ADA_ROOT = Path('/opt/minimax-h3/sage-sm89')


def package_path(capability, root=ADA_ROOT):
    if tuple(capability) != (8, 9):
        return ''
    if not (root / 'sageattention' / '__init__.py').is_file():
        raise RuntimeError('The SageAttention SM89 runtime is missing; refusing an unvalidated Ada fallback')
    return str(root)


if __name__ == '__main__':
    import torch
    print(package_path(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else '')
