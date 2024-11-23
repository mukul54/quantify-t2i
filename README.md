# QuantifyT2I

**Major contributions of this paper can be summarized as follows:**

1. We systematically identify and analyze the shortcomings of current T2I models in handling image generation tasks that require quantitative reasoning, emphasizing the importance of addressing these limitations for practical applications.
2. We introduce a novel approach that combines RLHF in diffusion models with curriculum learning and function calling to improve quantitative reasoning in T2I models.
3. We plan to conduct extensive experiments using established benchmarks and our own benchmark, demonstrating significant improvements in quantitative accuracy while maintaining image quality and image-text alignment.


## Fixed cached download issue in diffuser
`sed -i 's/from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info/from huggingface_hub import HfFolder, hf_hub_download, model_info/' ~/.conda/envs/d3po/lib/python3.10/site-packages/diffusers/utils/dynamic_modules_utils.py`