# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from .heatmap import Heatmap
from .road import Road
__all__ = ['Heatmap', 'get_dataset','Road']


def get_dataset(config):
    if config.DATASET.DATASET == 'heatmap':
        return Heatmap
    elif config.DATASET.DATASET == 'road':
        return Road
    else:
        print(config.DATASET.DATASET)
        raise NotImplemented()

