import gc
import torch

def clean_gpu_mem():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (torch.is_tensor(obj.data) and hasattr(obj, 'data')):
                del obj
        except:
            continue
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()