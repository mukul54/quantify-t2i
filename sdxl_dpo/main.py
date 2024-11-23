import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.resolve()
sys.path.append(str(src_path))

from src.trainer import train_dpo

if __name__ == "__main__":
    train_dpo()
