import numpy as onp
import jax.numpy as np
from torch.utils import data
from torchvision.datasets import MNIST

def numpy_collate(batch):
    if isinstance(batch[0], onp.ndarray):
        return onp.stack(batch)
    if isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return onp.array(batch)

class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super().__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
    def __call__(self, pic):
        return onp.ravel(onp.array(pic, dtype=np.float32))

def get_mnist_dataset(train):
    if train:
        mnist_dataset = MNIST('/tmp/mnist', download=True, transform=FlattenAndCast())
    else:
        mnist_dataset = MNIST('/tmp/mnist', download=True, train=False)
    return mnist_dataset

