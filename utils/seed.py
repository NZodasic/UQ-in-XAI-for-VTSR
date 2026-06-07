import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42, deterministic: bool = True, cudnn_benchmark: bool = False):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Configures CuDNN for either reproducibility or faster fixed-size training.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark and not deterministic)

def worker_init_fn(worker_id):
    """
    Worker init function for DataLoader to ensure fully reproducible behavior.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)
