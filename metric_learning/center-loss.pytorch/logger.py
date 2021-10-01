import os
import numpy as np
from tqdm import tqdm
from tqdmX import TqdmWrapper


#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from collections import deque, defaultdict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, max_len=3000):
        self.reset()
        self.val_deque = deque(maxlen=max_len)
        self.avg_deque = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val is None:
            return

        if np.isinf(np.abs(val)) or np.isnan(val):
            return

        self.val_deque.append(val)
        self.avg_deque = np.mean(self.val_deque)

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger:
    def __init__(self, pattern, folder, max_len=300, save_for_epoch=True, num_iterations=None):
        self.num_iterations = None
        self.save_for_epoch = save_for_epoch
        self.pattern = pattern
        self.folder = folder
        self.max_len = max_len

        self.folders = {
            'TRAIN': os.path.join(self.folder, 'TRAIN'),
            'VAL': os.path.join(self.folder, 'VAL')
        }
        os.makedirs(self.folders['TRAIN'], exist_ok=True)
        os.makedirs(self.folders['VAL'], exist_ok=True)

        self.writers = {
            'TRAIN': SummaryWriter(self.folders['TRAIN']),
            'VAL': SummaryWriter(self.folders['VAL'])
        }

        self.clear()

    def clear(self):
        self.losses_avg = {
            'TRAIN': defaultdict(lambda: AverageMeter(self.max_len)),
            'VAL': defaultdict(lambda: AverageMeter(self.max_len))
        }

        self.metrics_avg = {
            'TRAIN': defaultdict(lambda: AverageMeter(self.max_len)),
            'VAL': defaultdict(lambda: AverageMeter(self.max_len))
        }

    def __update(self, split, bar, epoch, iteration, losses, metrics, show_bar=True, write_log=True):
        # Update avg
        for k, v in losses.items():
            self.losses_avg[split][k].update(v)
        for k, v in metrics.items():
            self.metrics_avg[split][k].update(v)

        if show_bar:
            if isinstance(bar, TqdmWrapper):
                bar.add({k: round(v.avg_deque, 2) for k, v in self.losses_avg[split].items()})
                bar.add({k: round(v.avg_deque, 2) for k, v in self.metrics_avg[split].items()})
                bar.update()
            else:
                bar.set_description(self.pattern[split].format(epoch=epoch, 
                    losses=str({k: round(v.avg_deque, 2) for k, v in self.losses_avg[split].items()}), 
                    metrics=str({k: round(v.avg_deque, 2) for k, v in self.metrics_avg[split].items()})))

        # Writters
        if write_log:
            # epoch
            if self.save_for_epoch:
                for k, v in self.losses_avg[split].items():
                    self.writers[split].add_scalar('[loss] ' + k, v.avg, epoch)

                for k, v in self.metrics_avg[split].items():
                    self.writers[split].add_scalar('[metrics] ' + k, v.avg, epoch)
            else:
                # iteration
                for k, v in losses.items():
                    self.writers[split].add_scalar('[loss] ' + k, v, int((self.num_iterations * epoch) + iteration))

                for k, v in metrics.items():
                    self.writers[split].add_scalar('[metrics] ' + k, v, int((self.num_iterations * epoch) + iteration))

    def show_bar_train(self, bar, epoch, losses, metrics):
        return self.__update('TRAIN', bar, epoch, iteration=0, losses=losses, metrics=metrics, show_bar=True, write_log=False)

    def show_bar_test(self, bar, epoch, losses, metrics):
        return self.__update('VAL', bar, epoch, iteration=0, losses=losses, metrics=metrics, show_bar=True, write_log=False)

    def write_train_epoch(self, bar, epoch):
        return self.__update('TRAIN', bar, epoch, iteration=0, losses={}, metrics={}, show_bar=True, write_log=True)

    def write_test_epoch(self, bar, epoch):
        return self.__update('VAL', bar, epoch, iteration=0, losses={}, metrics={}, show_bar=True, write_log=True)