"""Allocate readable output names across ephemeral workers on the shared volume."""
import os
import re
import shutil
import uuid
from pathlib import Path


def copy_flat_output(source: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    # ComfyUI adds a worker-local counter; old delivery added a UUID after it.
    match = re.fullmatch(r'(.+?)_(\d+)_?(?:_[0-9a-f]{32})?', source.stem)
    prefix = match.group(1) if match else source.stem
    extension = source.suffix.lower()
    reservations = directory.parent / '.output-sequences' / directory.name / (prefix + extension)
    reservations.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(re.escape(prefix) + r'_(\d+)_?(?:_[0-9a-f]{32})?' + re.escape(extension))
    highest = 0
    for item in directory.iterdir():
        old = pattern.fullmatch(item.name)
        if old:
            highest = max(highest, int(old.group(1)))
    for item in reservations.iterdir():
        if item.name.isdigit():
            highest = max(highest, int(item.name))
    number = highest + 1
    while True:
        try:
            # Atomic mkdir on the shared filesystem arbitrates concurrent workers.
            (reservations / str(number)).mkdir()
        except FileExistsError:
            number += 1
            continue
        target = directory / f'{prefix}_{number:04d}{extension}'
        if target.exists():
            number += 1
            continue
        break
    temporary = directory / f'.{target.name}.{uuid.uuid4().hex}.partial'
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
