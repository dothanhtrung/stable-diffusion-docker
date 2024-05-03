#!/usr/bin/env bash
 
mkdir -p /workspace/ComfyUI/custom_nodes
cd /workspace/ComfyUI/custom_nodes
test -d comfyui_controlnet_aux || git clone https://github.com/Fannovel16/comfyui_controlnet_aux &
test -d ComfyUI_IPAdapter_plus || git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus &
test -d ComfyUI_UltimateSDUpscale || git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale &
test -d comfyui-inpaint-nodes || git clone https://github.com/Acly/comfyui-inpaint-nodes &
test -d comfyui-tooling-nodes || git clone https://github.com/Acly/comfyui-tooling-nodes &

mkdir -p /workspace/ComfyUI/custom_nodes/models/clip_vision/SD1.5
cd /workspace/ComfyUI/custom_nodes/models/clip_vision/SD1.5
test -f model.safetensors || wget "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors?download=true" -O model.safetensors &

mkdir -p /workspace/ComfyUI/custom_nodes/models/upscale_models
cd /workspace/ComfyUI/custom_nodes/models/upscale_models
test -f 4x_NMKD-Superscale-SP_178000_G.pth || wget "https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth" &
test -f OmniSR_X2_DIV2K.safetensors || wget "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X2_DIV2K.safetensors" &
test -f OmniSR_X3_DIV2K.safetensors || wget "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X3_DIV2K.safetensors" &
test -f OmniSR_X4_DIV2K.safetensors || wget "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X4_DIV2K.safetensors" &

mkdir -p /workspace/ComfyUI/custom_nodes/models/inpaint
cd /workspace/ComfyUI/custom_nodes/models/inpaint
test -f MAT_Places512_G_fp16.safetensors || wget "https://huggingface.co/Acly/MAT/resolve/main/MAT_Places512_G_fp16.safetensors" &
test -f fooocus_inpaint_head.pth || wget "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth" &
test -f inpaint_v26.fooocus.patch || wget "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch" &

mkdir -p /workspace/ComfyUI/custom_nodes/models/controlnet
cd /workspace/ComfyUI/custom_nodes/models/controlnet
test -f control_v11p_sd15_inpaint_fp16.safetensors || wget "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors" &
test -f control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors || wget "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors" &

mkdir -p /workspace/ComfyUI/custom_nodes/models/ipadapter
cd /workspace/ComfyUI/custom_nodes/models/ipadapter
test -f ip-adapter_sd15.safetensors || wget "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors" &
test -f ip-adapter_sdxl_vit-h.safetensors || wget "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors" &

mkdir -p /workspace/ComfyUI/custom_nodes/models/loras
cd /workspace/ComfyUI/custom_nodes/models/loras
test -f lcm-lora-sdv1-5.safetensors || wget "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors?download=true" -O lcm-lora-sdv1-5.safetensors &
test -f lcm-lora-sdxl.safetensors || wget "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors?download=true" -O lcm-lora-sdxl.safetensors &

# cd /workspace/stable-diffusion-webui/models/Stable-diffusion
# test -f ponyDiffusionV6XL_v6StartWithThisOne.safetensors || wget "https://civitai.com/api/download/models/290640?type=Model&format=SafeTensor&size=pruned&fp=fp16" -O ponyDiffusionV6XL_v6StartWithThisOne.safetensors &
# test -f realponycutejpFixed_No03.safetensors || wget "https://civitai.com/api/download/models/445811?type=Model&format=SafeTensor&size=full&fp=fp16" -O realponycutejpFixed_No03.safetensors &

# cd /workspace/stable-diffusion-webui/models/VAE
# test -f PonyDiffusionV6XL-VAE.safetensors || wget "https://civitai.com/api/download/models/290640?type=VAE&format=SafeTensor" -O PonyDiffusionV6XL-VAE.safetensors &

# rclone sync someonepod:/workspace/stable-diffusion-webui/models/Stable-diffusion /workspace/stable-diffusion-webui/models/Stable-diffusion
# rclone sync someonepod:/workspace/ComfyUI/models /workspace/ComfyUI/models
# rclone sync someonepod:/workspace/ComfyUI/custom_nodes /workspace/ComfyUI/custom_nodes
# rclone sync someonepod:/workspace/ComfyUI/extra_model_paths.yaml /workspace/ComfyUI/