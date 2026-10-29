# H3 acceleration recipes

Custom is the default and preserves the saved sampler, model and LoRAs, including
LCM. Turbo/Larry controls retain their existing behavior. Selecting a recipe is
an explicit model/schedule change; selecting Custom restores the previous values.

- PDD: FL2V/T2V or Ref2V stock INT8 checkpoint, original Alibaba 8-step
  checkpoint with the full displacement-head bank, dedicated H3PDDAcceleration
  node, exact eight-NFE sigmas, Euler and 12/3 shifts. Separate Turbo, sparse
  attention and FirstBlockCache are disabled. Creative LoRAs remain available;
  disable other acceleration adapters before selecting this recipe.
- SLA: FL2V/T2V stock INT8 checkpoint, converted LightX2V SLA adapter at strength
  one, native BlockSparseAttention, Euler four steps, 6/3 shifts and 15% exact
  blocks. The installed weights do not support Ref2V.
- Sol: available through the existing Attention workflow selector, using the
  native Comfy Kitchen kernel. No model adapter download is required.
- VSA: unavailable until compatible FastH3 weights are supplied. A selector alone
  does not make dense-trained weights compatible.

The existing learned latent upscale and RTX stages remain enabled as configured.
PDD completes its exact first pass, then uses the base model without PDD heads
for a separate low-noise refinement pass. Remaining-first-pass sigmas are
incompatible with these recipes. Prompt Book supports PDD and Sol; SLA is FL2V only.

PDD source is vendored under custom_nodes/h3_pdd at commit
c5f103aeafca90551f833b3a8a941776455d79e7 with its MIT license and two support
grids. It uses the runtime's existing Torch/ComfyUI dependencies. The image build
imports the extension on CPU and fails if it cannot load. No local CUDA package
or stopped pod restart is needed for installation.

Pinned assets (stored under models/loras/H3):

- lightx2v/Minimax-h3-Turbo-SLA at b4cd41131be06909110464ec1be3b30aa6068e8e:
  minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors
  SHA256 5cae6df40a06ea825f85fc8876c9ea1c9692c833a9af07bb8b3bac9ce2a71bac
- alibaba-pai/MiniMax-H3-Acc-LoRAs at 335001fb9e5455d68a0caa18ec2e319072150328:
  MiniMax-H3-FL2VA-Acc-8Step.safetensors
  SHA256 0b29be7042d883970eb0c20774a9ba03d95669ed80a721bb4d21be8ea0d0a196
  and MiniMax-H3-Ref2VA-Acc-8Step.safetensors
  SHA256 111c82e669f6e20e628228172edf39395f1a9fc3ad049793895e542c0f55b18c

References:
https://github.com/lukas-9936/ComfyUI-MiniMax-H3-PDD
https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs
https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA
https://docs.comfy.org/tutorials/video/minimax/minimax-h3
