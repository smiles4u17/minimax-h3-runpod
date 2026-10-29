"""ComfyUI loader for Alibaba PAI's MiniMax-H3 PDD acceleration LoRAs."""

import logging

log = logging.getLogger("H3PDD")

_EXTENSION = None
try:
    from comfy_api.latest import ComfyExtension

    from .adaln_node import H3PDDStandaloneAdaLNFix
    from .pdd_node import H3PDDAcceleration

    class H3PDDExtension(ComfyExtension):
        async def get_node_list(self):
            return [H3PDDAcceleration, H3PDDStandaloneAdaLNFix]

    _EXTENSION = H3PDDExtension
except Exception:  # noqa: BLE001 - never prevent ComfyUI from starting
    log.exception("[H3PDD] failed to load; the node will not be available. "
                  "ComfyUI startup is unaffected.")

if _EXTENSION is not None:
    async def comfy_entrypoint():
        return _EXTENSION()

    __all__ = ["comfy_entrypoint"]
else:
    __all__ = []
