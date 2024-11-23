import torch
from typing import Tuple
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from transformers import AutoTokenizer
from peft import LoraConfig
from config.training_config import TrainingConfig

def load_models(config: TrainingConfig) -> Tuple[AutoencoderKL, UNet2DConditionModel, CLIPTextModel, CLIPTextModelWithProjection]:
    """Load and configure all required models"""
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path, 
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    
    # Create reference UNet
    ref_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32
    )
    # Text encoders
    text_encoder_one = CLIPTextModel.from_pretrained(
        config.pretrained_model_name_or_path, 
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        config.pretrained_model_name_or_path, 
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    return vae, unet, ref_unet, text_encoder_one, text_encoder_two, tokenizer

def configure_models(
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    ref_unet:UNet2DConditionModel,
    text_encoder_one: CLIPTextModel,
    text_encoder_two: CLIPTextModelWithProjection,
    config: TrainingConfig,
    device: torch.device
) -> None:
    """Configure models with memory optimizations"""
    # Enable memory efficient attention
    if config.enable_xformers:
        unet.enable_xformers_memory_efficient_attention()
        ref_unet.enable_xformers_memory_efficient_attention()
        vae.enable_xformers_memory_efficient_attention()
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        ref_unet.enable_gradient_checkpointing()

    # Move models to device

    vae.to(device)
    text_encoder_one.to(device)
    text_encoder_two.to(device)
    unet.to(device)
    ref_unet.to(device)
    
    # Freeze and set to eval mode
    for model in [vae, ref_unet, text_encoder_one, text_encoder_two]:
        model.requires_grad_(False)
        model.eval()

def add_lora_to_unet(unet: UNet2DConditionModel, config: TrainingConfig) -> None:
    """Add LoRA adapter to UNet"""
    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)