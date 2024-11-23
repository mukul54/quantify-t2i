from dataclasses import dataclass, field
from typing import List
from typing import Optional

def get_default_prompts() -> List[str]:
    return [
        "Generate an image with exactly eleven pencils arranged neatly on a desk.",
        "Create an image of exactly fifteen bicycles parked in a row outside a library."
    ]

@dataclass
class TrainingConfig:
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    output_dir: str = "/l/users/mukul.ranjan/sdxl_dp/ckpt/sdxl-dpo-lora-output"
    resolution: int = 1024
    wandb_project: str = "sdxl-dpo-lora"
    sample_prompts: List[str] = field(default_factory=get_default_prompts)
    
    # Memory optimization parameters
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4 # Start without accumulation: 1
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = False
    chunk_size: int = 4  # Added this line for VAE batch processing
    
    # Training parameters
    learning_rate = 5e-5  # Lower learning rate
    beta_dpo = 100.0  # Lower DPO weight initially

    max_train_steps: int = 20000
    checkpointing_steps: int = 2000
    rank: int = 16
    seed: int = 42
    validation_steps: int = 100
    enable_xformers: bool = True
    vae_encode_batch_size: int = 2
    
    # Paths
    image_dir: str = "/home/mukul.ranjan/projects/sdxl_dpo/data/images"
    human_feedback_path: str = "/home/mukul.ranjan/projects/sdxl_dpo/data/hf/human_feedback.json"
    # Scheduler parameters
    lr_scheduler_type: str = "cosine"  # Options: "linear", "cosine", "cosine_with_restarts", "polynomial"
    warmup_ratio: float = 0.1  # Percentage of steps to use for warmup
    min_lr_ratio: float = 0.1  # Minimum learning rate as a ratio of initial lr
    num_warmup_steps: Optional[int] = None  # If None, will be calculated from warmup_ratio
    
    def get_warmup_steps(self) -> int:
        """Calculate warmup steps based on ratio or return specified steps"""
        if self.num_warmup_steps is not None:
            return self.num_warmup_steps
        return min(int(self.max_train_steps * self.warmup_ratio), 1000)
    
    def get_min_lr(self) -> float:
        """Calculate minimum learning rate"""
        return self.learning_rate * self.min_lr_ratio
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.train_batch_size > 1:
            print("Warning: train_batch_size > 1 might cause memory issues")
        if not self.enable_xformers:
            print("Warning: xformers is disabled, this might lead to higher memory usage")
