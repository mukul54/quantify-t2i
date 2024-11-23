# QuantifyT2I

This repository contains the implementation for our paper "Quantitative Reasoning for Text-to-Image Generation", focusing on enhancing quantitative reasoning capabilities in text-to-image models.

## Key Contributions

1. A comprehensive analysis of quantitative reasoning limitations in current T2I models through systematic evaluation on counting accuracy, size proportionality, and fractional understanding.

2. SDXL-DPO: A novel fine-tuning approach combining Direct Preference Optimization with LoRA adaptation for improved quantitative reasoning in T2I models.

3. A curated evaluation framework including automated metrics and human assessment, demonstrating significant improvements in numerical understanding while maintaining image quality.

## Repository Structure
```
.
├── annotation_tool/         # Interface for human evaluation
├── automated_evaluation/    # Scripts for automated metrics
├── LLM-groundedDiffusion/  # Implementation of LLM-guided baseline
├── make-it-count/          # Implementation of Make-It-Count baseline
├── sdxl_dpo/               # Our SDXL-DPO implementation
└── src/                    # Core utilities and data processing
```

## Setup and Training

### SDXL-DPO Training
```bash
cd sdxl_dpo
pip install -r requirements.txt
python main.py
```

### Image Generation
```bash
python generate_image.py
```

### Other Methods
- For LLM-Grounded Diffusion setup and usage, see [LLM-groundedDiffusion/README.md](LLM-groundedDiffusion/README.md)
- For Make-It-Count implementation details, see [make-it-count/README.md](make-it-count/README.md)

## Requirements

### Hardware Requirements
- SDXL-DPO: Single NVIDIA A100 GPU (40GB)
- LLM-Grounded Diffusion: Check respective README
- Make-It-Count: Check respective README

### Software Requirements
For SDXL-DPO:
```bash
torch>=2.0.0
diffusers>=0.24.0
transformers>=4.35.0
accelerate>=0.24.0
peft>=0.6.0
wandb
```

## Evaluation

### Automated Evaluation
```bash
cd automated_evaluation
python eval_pipeline.py
```

### Human Evaluation Interface
```bash
cd annotation_tool
python t2i_interface.py
```

## Dataset
The curated dataset of 350 prompt-image pairs for quantitative reasoning evaluation is available at [dataset link]

The dataset includes:
- Counting tasks (150 pairs)
- Size proportionality (100 pairs)
- Fractional understanding (100 pairs)

## Model Weights
Pre-trained model weights for SDXL-DPO will be released upon publication.

## Citation
```bibtex
[Citation will be added upon publication]
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- SDXL base model from Stability AI
- Evaluation framework adapted from [GECKO-NUM](https://github.com/google-deepmind/geckonum_benchmark_t2i/tree/main)
- make-it-count from [make-it-count](https://github.com/Litalby1/make-it-count/tree/main/train_relayout)
- LLM-GroundedDiffusion[LLM-GroundedDiffusion](https://github.com/TonyLianLong/LLM-groundedDiffusion)
- Diffusion-DPO[Diffusion-DPO](https://github.com/SalesforceAIResearch/DiffusionDPO?tab=readme-ov-file)