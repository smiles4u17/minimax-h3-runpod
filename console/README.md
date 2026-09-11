# RunPod Media Console Web

Local browser UI for preparing media, running RunPod InfiniteTalk/WanAnimate jobs, creating SAM3 masks, and managing outputs.

Start the app:

```powershell
.\run_windows.bat
```

Then open:

```text
http://127.0.0.1:8765
```

The default run scripts are private to this PC. To use another device on your local network, start `run_local_network.bat`, then open:

```text
http://YOUR-PC-LAN-IP:8765
```

The LAN script prints a generated username/password. Remote browsers must use those credentials; requests from this PC remain password-free. If Windows Firewall prompts, allow access on Private networks only.

Current app version: `web-v15.51-h3-sage-quality-baseline`

## Main Workflows

- **Workflow**: guided subject-replacement flow using SAMimate.
- **InfiniteTalk**: image/video plus audio talking-head submissions.
- **WanAnimate**: reference image plus driving video animation/replacement jobs.
- **MiniMax H3**: FL2V first/last-frame and R2V reference-image/audio jobs, H3-volume LoRAs, live network-volume diffusion model/CLIP/VAE dropdowns, genuine Turbo on/off controls, preset/custom aspect ratios, clean quality baselines, and an H3-volume ComfyUI Pod launcher.
- **H3 Model Manager**: background multipart downloads from Hugging Face, Civitai, or direct HTTPS URLs into checkpoints, diffusion models, CLIP/text encoders, VAEs, LoRAs, ControlNet, embeddings, and upscale-model folders on the H3 volume.
- **SAMimate**: local SAM3 mask generation, WanAnimate submission, and final composite.
- **SAM Studio**: local SAM3 mask generation and preview.
- **Text to Speech**: free local Windows SAPI or Piper TTS generation.
- **Media Browser**: local/S3 browsing with Main/H3 bucket switching, preview, selection, downloads, sorting, and media repair.
- **Settings**: RunPod, S3, output/cache, SAM3, Piper, and endpoint profile setup.

## Documentation

- [User Guide](docs/USER_GUIDE.md)
- [Workflow Reference](docs/WORKFLOWS.md)
- [Settings Reference](docs/SETTINGS.md)
- [Tooltip Reference](docs/TOOLTIPS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Storage

- Local settings: `%USERPROFILE%\.runpod_media_console_web`
- Default app cache: `D:\RunPodMediaConsoleWeb\cache`
- Default outputs: `D:\RunPodMediaConsoleWeb\outputs`
- Default SAM outputs: `D:\RunPodMediaConsoleWeb\sam_outputs`

MiniMax H3 completed videos are saved under `outputs\h3`. Its endpoint and network-volume S3 routing are independent from the global endpoint storage settings.

Worker outputs are resolved from their exact nested S3 key, normally `output/h3/<job-id>/<filename>`. The fallback downloader stays inside the current job folder so a missed completion cannot pull an older generation.

H3 LoRAs are stored on that dedicated volume under `/runpod-volume/models/loras/H3`. Uploading or importing a LoRA automatically refreshes the library and adds it to the current workflow with an editable strength from 0 to 2.

Use **H3 Model Manager** for larger or non-LoRA files. Hugging Face file/blob URLs and Civitai model/version/download URLs are supported; optional provider tokens are saved as redacted secrets. Downloads stream directly into multipart S3 storage and expose progress without staging the checkpoint on the local disk.

Use the Media Browser's Bucket selector to view either the original Main bucket or the newer H3 bucket. **Use H3 bucket as Main** changes future global routing but does not copy old objects; attach that volume to other endpoints before using its volume paths there.

The **H3 ComfyUI** control reuses a matching Pod when one exists. Starting or creating a Pod requires an explicit billing confirmation. Before launch, the app merges `/workspace/models` into the official ComfyUI `extra_model_paths.yaml`, preserving existing entries.

Cache cleanup only removes app cache files. It does not remove final outputs, SAM outputs, or Hugging Face model weights.
