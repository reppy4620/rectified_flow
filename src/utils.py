import os
import random
import numpy as np
import torch
from collections import defaultdict
from torchvision.utils import make_grid
from torchvision import transforms as T


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_grid(image_tensors, output_path, nrow=4):
    img = make_grid(image_tensors, nrow=nrow, normalize=True)
    img = T.ToPILImage()(img)
    img.save(output_path)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def mean(self):
        return self.avg


class LossTracker(defaultdict):
    def __init__(self):
        super().__init__(AverageMeter)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k].update(v)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

class Tracker:
    def __init__(self, log_file, mode='w'):
        super().__init__()
        self.loss_tracker = LossTracker()
        self.log_file = log_file
        self.mode = mode

    def update(self, **kwargs):
        self.loss_tracker.update(**kwargs)

    def __getattr__(self, key):
        return self.loss_tracker.get(key)

    def __getitem__(self, key):
        return self.loss_tracker.get(key)
    
    def write(self, epoch, clear=True):
        with open(self.log_file, self.mode) as f:
            if epoch == 1:
                f.write(','.join(['epoch'] + list(self.loss_tracker.keys()))+'\n')
                f.write(','.join([str(epoch)]+[str(v.mean()) for v in self.loss_tracker.values()])+'\n')
            else:
                f.write(','.join([str(epoch)]+[str(v.mean()) for v in self.loss_tracker.values()])+'\n')

        if clear:
            self.loss_tracker.clear()
