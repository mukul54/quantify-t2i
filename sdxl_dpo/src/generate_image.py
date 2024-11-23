import json
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from safetensors.torch import load_file
from pathlib import Path
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
lora_path = Path("/l/users/mukul.ranjan/sdxl_dp/ckpt/sdxl-dpo-lora-output/checkpoint-8000/pytorch_lora_weights.safetensors")
prompts_file_path = "prompts_geckonum.json" # "prompts.json"
generated_dir_name = "generated_images_geckonum_8k"

def load_prompts(prompt_file):
    """Load prompts from JSON file"""
    with open(prompt_file, 'r') as f:
        prompts = json.load(f)
    return prompts

def setup_pipeline():
    """Setup SDXL pipeline with LoRA weights"""
    # Load base model
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to("cuda:0")

    # Load LoRA weights

    state_dict = load_file(lora_path)
    
    # Load and fuse LoRA weights
    pipeline.unet.load_state_dict(state_dict, strict=False)
    
    # Enable memory optimizations
    pipeline.enable_model_cpu_offload()
    
    return pipeline

def generate_images(pipeline, prompts, output_dir="generated_images_ckpt_8k", num_images_per_prompt=1):
    """Generate images for all prompts"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save prompts for reference
    with open(output_dir / "prompts.json", 'w') as f:
        json.dump(prompts, f, indent=2)
    
    for idx, prompt in enumerate(prompts):
        try:
            print(f"\nGenerating image {idx + 1}/{len(prompts)}")
            print(f"Prompt: {prompt}")
            
            # Generate image
            images = pipeline(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                num_images_per_prompt=num_images_per_prompt
            ).images
            
            # Save images
            for img_idx, image in enumerate(images):
                image_path = output_dir / f"image_{idx:04d}_{img_idx}.png"
                image.save(image_path)
                print(f"Saved to {image_path}")
            
            # Optional: Add small delay between generations
            time.sleep(1)

        except Exception as e:
            print(f"Error generating image for prompt {idx}: {e}")
            continue
        
        # Clear GPU memory
        torch.cuda.empty_cache()

def main():

    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    # Load prompts
    prompts = load_prompts(prompts_file_path)
    print(f"Loaded {len(prompts)} prompts")
    
    # Setup pipeline
    print("Setting up pipeline...")
    pipeline = setup_pipeline()
    
    # Generate images
    print("Starting generation...")
    generate_images(pipeline, output_dir = generated_dir_name, prompts=prompts, num_images_per_prompt=1)
    
    print("Generation complete!")

if __name__ == "__main__":
    main()