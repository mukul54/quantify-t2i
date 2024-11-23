# src/utils/memory_utils.py

import psutil
import torch
import gc
from typing import Dict

def get_memory_stats() -> Dict[str, float]:
    """Get current memory usage statistics"""
    memory_stats = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": psutil.virtual_memory().used / (1024**3),
    }
    
    if torch.cuda.is_available():
        memory_stats.update({
            "gpu_memory_used": torch.cuda.memory_allocated() / (1024**3),
            "gpu_memory_cached": torch.cuda.memory_reserved() / (1024**3),
        })
    
    return memory_stats

def clear_memory():
    """Clear unused memory"""
    gc.collect()
    torch.cuda.empty_cache()
    
    if hasattr(torch.cuda, 'memory_summary'):
        print(torch.cuda.memory_summary())