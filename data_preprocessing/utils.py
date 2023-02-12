import numpy as np

import inspect
from datetime import datetime


def load_default_identifiers(n, g, l):
    if n is None:
        n = 'features'
    if g is None:
        g = 'graph'
    if l is None:
        l = 'label'
    return n, g, l


def initialize_batch(entries, batch_size, shuffle=False):
    total = len(entries)
    indices = np.arange(0, total - 1, 1)
    if shuffle:
        np.random.shuffle(indices)
    batch_indices = []
    start = 0
    end = len(indices)
    curr = start
    while curr < end:
        c_end = curr + batch_size
        if c_end > end:
            c_end = end
        batch_indices.append(indices[curr:c_end])
        curr = c_end
    return batch_indices[::-1]