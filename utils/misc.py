import time
import logging
import matplotlib.pyplot as plt
import numpy as np

def timeit(f):
    """ Decorator to time Any Function """

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        logging.getLogger("Timer").info("   [-] %s : %2.5f sec, which is %2.5f min, which is %2.5f hour" %
                                        (f.__name__, seconds, seconds / 60, seconds / 3600))
        return result

    return timed


def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")
    import sys
    from subprocess import call
    import torch
    logger.info('__Python VERSION:  {}'.format(sys.version))
    logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
    logger.info('__CUDA VERSION')
    call(["nvcc", "--version"])
    logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.info('Available devices  {}'.format(torch.cuda.device_count()))
    logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated