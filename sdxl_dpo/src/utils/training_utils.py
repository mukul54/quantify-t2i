# src/utils/trainin_utils.py
import torch
from typing import Tuple
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from peft import LoraConfig
from config.training_config import TrainingConfig

import torch
import wandb
import numpy as np
from typing import List, Optional
from diffusers import DiffusionPipeline
from torchvision.utils import make_grid
from config.training_config import TrainingConfig

def print_gpu_memory() -> None:
    """Print current GPU memory usage"""
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1024**2:.2f}MB")

def generate_samples(
    pipeline: DiffusionPipeline,
    prompts: List[str],
    num_inference_steps: int = 30,
    seed: Optional[int] = None
) -> torch.Tensor:
    """Generate sample images using the pipeline with memory optimization"""
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    else:
        generator = None
        
    images = []
    for prompt in prompts:
        try:
            with torch.autocast("cuda"), torch.no_grad():
                image = pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images[0]
                images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
                
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")
            continue
    
    return make_grid(images, nrow=2)

def log_predictions(
    pipeline: DiffusionPipeline,
    args: TrainingConfig,
    step: int,
) -> None:
    """Generate and log sample predictions with memory management"""
    pipeline.set_progress_bar_config(disable=True)
    
    try:
        # Generate samples
        with torch.no_grad():
            sample_images = generate_samples(
                pipeline=pipeline,
                prompts=args.sample_prompts,
                seed=args.seed,
            )
        
        # Log to wandb
        wandb.log(
            {
                "samples": wandb.Image(sample_images),
                "global_step": step,
            },
            step=step,
        )
        
    except Exception as e:
        print(f"Error in log_predictions: {e}")
    
    finally:
        torch.cuda.empty_cache()

def get_optimizer(
    model_params,
    config: TrainingConfig
) -> torch.optim.Optimizer:
    """Get optimizer with optional 8-bit precision"""
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            print("Using 8-bit Adam optimizer")
        except ImportError:
            print("bitsandbytes not found, using regular AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW
        
    return optimizer_cls(
        model_params,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

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
    
    return vae, unet, text_encoder_one, text_encoder_two

def configure_models(
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    text_encoder_one: CLIPTextModel,
    text_encoder_two: CLIPTextModelWithProjection,
    config: TrainingConfig,
    device: torch.device
) -> None:
    """Configure models with memory optimizations"""
    # Enable memory efficient attention
    if config.enable_xformers:
        unet.enable_xformers_memory_efficient_attention()
        vae.enable_xformers_memory_efficient_attention()
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # Move models to device
    vae.to(device)
    text_encoder_one.to(device)
    text_encoder_two.to(device)
    
    # Freeze and set to eval mode
    for model in [vae, text_encoder_one, text_encoder_two]:
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