# src/trainer.py
import os
import wandb
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler, DiffusionPipeline
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel

from config.training_config import TrainingConfig
from src.data.dataset import DPODataset, create_dataloader
from src.models.model_utils import load_models, configure_models, add_lora_to_unet
from src.utils.training_utils import print_gpu_memory, log_predictions, get_optimizer


class DPOTrainer:
    def __init__(self, config: TrainingConfig):
        
        self.config = config
        self.global_step = 0
        self.setup_accelerator()
        self.setup_models()
        self.setup_data()
        self.setup_optimization()

    def setup_accelerator(self):
        self.accelerator = Accelerator(
            mixed_precision="fp16",  # Keep fp16 but handle gradients properly
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="wandb",
        )

    def setup_models(self):
        if self.accelerator.is_main_process:
            print("Loading models...")
            print_gpu_memory()
        
        # Load models
        self.vae, self.unet, self.ref_unet, self.text_encoder_one, self.text_encoder_two, self.tokenizer = load_models(self.config)
        
        
        # Configure models
        configure_models(
            self.vae, self.unet, self.ref_unet, self.text_encoder_one, self.text_encoder_two,
            self.config, self.accelerator.device
        )
        
        # Add LoRA and ensure UNet params are float32
        add_lora_to_unet(self.unet, self.config)
        
        # Important: Convert UNet's trainable parameters to float32
        for param in self.unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)

    def setup_data(self):
        dataset = DPODataset(
            image_dir=self.config.image_dir,
            human_feedback_path=self.config.human_feedback_path,
            resolution=self.config.resolution,
            use_fp16=True  # Use FP16
        )
        train_dataset = dataset.create_hf_dataset()
        self.train_dataloader = create_dataloader(
            dataset=train_dataset,
            batch_size=self.config.train_batch_size,
            num_workers=0,
            pin_memory=True,
        )

    def generate_samples(self):
        """Generate and log sample images to wandb"""
        print("Generating samples...")
        
        # Create pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            vae=self.vae,
            unet=self.accelerator.unwrap_model(self.unet),
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            scheduler=self.noise_scheduler,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        
        # Move to device and enable memory savings
        pipeline.to(self.accelerator.device)
        if self.config.enable_xformers:
            pipeline.enable_xformers_memory_efficient_attention()
        pipeline.set_progress_bar_config(disable=True)
        
        # Generate images
        images = []
        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)
        
        with torch.no_grad():
            for prompt in self.config.sample_prompts:
                try:
                    # Generate with proper settings
                    output = pipeline(
                        prompt=prompt,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        height=1024,
                        width=1024,
                        generator=generator,
                        output_type="np",
                    )
                    
                    # Process image
                    image = output.images[0]
                    image = np.clip(image, 0, 1)
                    image = (image * 255).round().astype("uint8")
                    image = Image.fromarray(image)
                    
                    images.append(
                        wandb.Image(
                            image, 
                            caption=f"Step {self.global_step}: {prompt}"
                        )
                    )
                    
                    print(f"Successfully generated image for prompt: {prompt}")
                    
                except Exception as e:
                    print(f"Error generating image for prompt '{prompt}': {e}")
                    continue
                
                torch.cuda.empty_cache()
        
        # Log to wandb
        if self.accelerator.is_main_process and images:
            wandb.log({
                "samples": images,
                "step": self.global_step
            })
            print(f"Successfully logged {len(images)} images to wandb")
        
        # Cleanup
        del pipeline
        torch.cuda.empty_cache()
        print("Sample generation complete")

        
    # In setup_optimization method:
    def setup_optimization(self):
        """Setup optimizer and schedulers with better stability"""
        # Setup noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, 
            subfolder="scheduler"
        )

        # Use a lower learning rate for stability
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config.learning_rate * 0.5,  # Halve the learning rate
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-6  # Slightly increased weight decay
        )

        # Learning rate scheduler with warmup
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=int(self.config.max_train_steps * 0.1),  # 10% warmup
            num_training_steps=self.config.max_train_steps
        )

        # Prepare with accelerator
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def compute_time_ids(self, original_size, crop_coords_top_left, target_size=(1024, 1024)):
        """
        Compute time ids following SDXL format
        Args:
            original_size: Tuple of (height, width)
            crop_coords_top_left: Tuple of (top, left) coordinates
            target_size: Tuple of (height, width) for target resolution
        """
        # Combine all required values in the correct order
        add_time_ids = list(original_size) + list(crop_coords_top_left) + list(target_size)
        add_time_ids = torch.tensor([add_time_ids], device=self.accelerator.device, dtype=torch.float32)
        return add_time_ids

    def process_vae_batch(self, images, original_sizes=None, crop_coords=None):
        """Process a batch of images through VAE with memory optimization"""
        try:
            # print(f"\nInput images shape: {images.shape}")
            
            # Reshape if needed
            if len(images.shape) == 2:  # If flattened
                # print("image dimension is flattened with 2-dimension")
                batch_size = images.shape[0] // (3 * 1024)
                images = images.view(batch_size, 3, 1024, 1024)
            elif len(images.shape) == 3:  # If missing batch dimension
                print("images missing batch dimension")
                images = images.unsqueeze(0)
            
            # print(f"Final shape before VAE: {images.shape}")
            
            # Store original sizes if not provided
            if original_sizes is None:
                original_sizes = [(images.shape[2], images.shape[3])] * images.shape[0]
            
            # Store crop coordinates if not provided
            if crop_coords is None:
                # For center crop, calculate the coordinates
                crop_coords = [(0, 0)] * images.shape[0]  # Default to top-left
            
            # Process through VAE
            with torch.no_grad():
                images = images.to(device=self.accelerator.device, dtype=torch.float16)
                latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
                # print(f"Latents shape: {latents.shape}")
                
                return latents, original_sizes, crop_coords
                
        except Exception as e:
            print(f"Error in process_vae_batch: {e}")
            print(f"Images type: {type(images)}")
            if isinstance(images, torch.Tensor):
                print(f"Images shape: {images.shape}")
                print(f"Images dtype: {images.dtype}")
            return None, None, None

    def compute_loss(self, pred, target):
        """Compute MSE loss with numeric stability"""
        # Handle potential NaN/Inf values
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e5, neginf=-1e5)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e5, neginf=-1e5)
        
        # Compute per-element MSE
        per_element_mse = (pred - target).pow(2)
        
        # Clip extreme values
        per_element_mse = torch.clamp(per_element_mse, min=0.0, max=1e5)
        
        # Mean across spatial dimensions
        return per_element_mse.mean(dim=[1, 2, 3])

    def train_step(self, batch):
        with self.accelerator.accumulate(self.unet):
            preferred_latents, preferred_sizes, preferred_crops = self.process_vae_batch(batch["preferred_image"])
            if preferred_latents is None:
                print("Failed to process preferred images")
                return None, None, None, None
                
            rejected_latents, rejected_sizes, rejected_crops = self.process_vae_batch(batch["rejected_image"])
            if rejected_latents is None:
                print("Failed to process rejected images")
                return None, None, None, None

            batch_size = preferred_latents.shape[0]
            target_size = (self.config.resolution, self.config.resolution)

            # Generate distinct time embeddings
            preferred_time_ids = torch.cat([
                self.compute_time_ids(size, crop, target_size) 
                for size, crop in zip(preferred_sizes, preferred_crops)
            ])
            
            rejected_time_ids = torch.cat([
                self.compute_time_ids(size, crop, target_size) 
                for size, crop in zip(rejected_sizes, rejected_crops)
            ])

            # Text encoder processing
            text_input_ids = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.accelerator.device)

            with torch.no_grad():
                encoder_output_1 = self.text_encoder_one(text_input_ids, output_hidden_states=True)
                encoder_output_2 = self.text_encoder_two(text_input_ids, output_hidden_states=True)
                
                text_embeddings_1 = encoder_output_1.hidden_states[-2]
                text_embeddings_2 = encoder_output_2.hidden_states[-2]
                pooled_embeddings = encoder_output_2.text_embeds
                text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2], dim=-1)

            # Concatenate latents for batch processing
            combined_latents = torch.cat([preferred_latents, rejected_latents])
            combined_time_ids = torch.cat([preferred_time_ids, rejected_time_ids])
            
            # Single noise for both
            noise = torch.randn_like(combined_latents)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (combined_latents.shape[0],), device=self.accelerator.device)
            noisy_latents = self.noise_scheduler.add_noise(combined_latents, noise, timesteps)

            # Get model predictions with gradient
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings.repeat(2, 1, 1),
                added_cond_kwargs={
                    "text_embeds": pooled_embeddings.repeat(2, 1),
                    "time_ids": combined_time_ids
                }
            ).sample

            # Get reference model predictions
            with torch.no_grad():
                ref_pred = self.ref_unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings.repeat(2, 1, 1),
                    added_cond_kwargs={
                        "text_embeds": pooled_embeddings.repeat(2, 1),
                        "time_ids": combined_time_ids
                    }
                ).sample

            # Calculate losses with numeric stability
            model_losses = self.compute_loss(model_pred, noise)
            model_losses_w, model_losses_l = model_losses.chunk(2)
            model_diff = model_losses_w - model_losses_l

            ref_losses = self.compute_loss(ref_pred, noise)
            ref_losses_w, ref_losses_l = ref_losses.chunk(2)
            ref_diff = ref_losses_w - ref_losses_l

            # Compute DPO loss with stabilization
            eps = 1e-8  # Small epsilon for numeric stability
            scale_term = -0.5 * self.config.beta_dpo
            
            # Clip differences to avoid extreme values
            model_diff = torch.clamp(model_diff, min=-1e3, max=1e3)
            ref_diff = torch.clamp(ref_diff, min=-1e3, max=1e3)
            
            loss_diff = scale_term * (model_diff - ref_diff)
            loss_diff = torch.clamp(loss_diff, min=-50.0, max=50.0)
            
            # Compute final loss with stability
            loss = -torch.log(torch.sigmoid(loss_diff) + eps)
            loss = torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=-1.0).mean()

            # Print debugging info
            print(f"model_diff: {model_diff.mean().item():.4f}")
            print(f"ref_diff: {ref_diff.mean().item():.4f}")
            print(f"loss_diff: {loss_diff.mean().item():.4f}")
            print(f"final_loss: {loss.item():.4f}")

            wandb.log({
                "debug/model_preferred_loss": model_losses_w.mean().item(),
                "debug/model_rejected_loss": model_losses_l.mean().item(),
                "debug/ref_preferred_loss": ref_losses_w.mean().item(),
                "debug/ref_rejected_loss": ref_losses_l.mean().item(),
                "debug/model_ratio": (model_losses_w/model_losses_l).mean().item(),
                "debug/ref_ratio": (ref_losses_w/ref_losses_l).mean().item(),
            }, step=self.global_step)
            
            # Skip step if loss is invalid
            if torch.isnan(loss) or torch.isinf(loss):
                print("Skipping step due to invalid loss")
                return None, None, None, None

            # Optimization steps
            self.accelerator.backward(loss)
            
            if self.accelerator.sync_gradients:
                # Clip gradients with lower threshold
                if any(p.grad is not None for p in self.unet.parameters()):
                    self.accelerator.clip_grad_norm_(
                        [p for p in self.unet.parameters() if p.requires_grad], 
                        0.5  # Reduced from 1.0
                    )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            return loss, model_diff.mean(), ref_diff.mean(), loss_diff.mean()

    def train(self):
        if self.accelerator.is_main_process:
            print("Starting training...")
            print_gpu_memory()
            
            wandb.watch(
                self.unet,
                log="all",
                log_freq=100,
                log_graph=True
            )
            
            self.generate_samples()

        for epoch in range(self.config.max_train_steps):
            for batch in self.train_dataloader:
                loss, model_diff, ref_diff, loss_diff = self.train_step(batch)
                
                if loss is None:
                    continue
                    
                if self.accelerator.is_main_process:
                    wandb.log({
                        "training/loss": loss.item(),
                        "training/model_diff": model_diff.item(),
                        "training/ref_diff": ref_diff.item(),
                        "training/loss_diff": loss_diff.item(),
                        "training/learning_rate": self.lr_scheduler.get_last_lr()[0]
                    }, step=self.global_step)
                    
                    if self.global_step > 0 and self.global_step % 500 == 0:
                        try:
                            self.generate_samples()
                        except Exception as e:
                            print(f"Error in sample generation: {e}")
                    
                    if self.global_step % self.config.checkpointing_steps == 0:
                        self.save_checkpoint(self.global_step)
                
                self.global_step += 1
                if self.global_step >= self.config.max_train_steps:
                    break
            
            if self.global_step >= self.config.max_train_steps:
                break

        self.accelerator.end_training()
        wandb.finish()

    def save_checkpoint(self, step):
        save_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)
        
        unet = self.accelerator.unwrap_model(self.unet)
        
        StableDiffusionXLLoraLoaderMixin.save_lora_weights(
            save_directory=save_dir,
            unet_lora_layers=unet.state_dict(),
            safe_serialization=True,
        )
        
        print(f"Saved checkpoint to {save_dir}")


def train_dpo():
    config = TrainingConfig()
    wandb.init(project=config.wandb_project, config=vars(config))
    trainer = DPOTrainer(config)
    trainer.train()
