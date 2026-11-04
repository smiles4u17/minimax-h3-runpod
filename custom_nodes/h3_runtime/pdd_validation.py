"""Write a small PDD setup receipt without running sampling or decoding."""
import json
from pathlib import Path
import folder_paths


class H3PDDValidationOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ()
    FUNCTION = "validate"
    OUTPUT_NODE = True
    CATEGORY = "video/minimax/diagnostics"

    def validate(self, model):
        marker = model.model_options.get("transformer_options", {}).get("_plaguekind_h3_pdd")
        if not marker or marker.get("nfe") != 8:
            raise ValueError("PDD validation requires the complete 8-step PDD model")
        curve = marker.get("curve_adaln")
        if curve and (curve.get("effective_mode") != "port" or curve.get("ported", 0) < 50):
            raise ValueError("PDD validation requires all curve AdaLN adapters")
        root, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            "validation/pdd_setup", folder_paths.get_output_directory())
        filename = f"{name}_{counter:05}_.json"
        Path(root).mkdir(parents=True, exist_ok=True)
        Path(root, filename).write_text(json.dumps({"validated": True, **marker}, indent=2))
        return {"ui": {"files": [{"filename": filename, "subfolder": subfolder, "type": "output"}]}}
