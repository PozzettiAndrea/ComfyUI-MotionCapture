import logging
import pickle
import os
import os.path as osp

# RGBD-Dataset
from .tartan import TartanAir

log = logging.getLogger("motioncapture")

def dataset_factory(dataset_list, **kwargs):
    """ create a combined dataset """

    from torch.utils.data import ConcatDataset

    dataset_map = { 
        'tartan': (TartanAir, ),
    }

    db_list = []
    for key in dataset_list:
        # cache datasets for faster future loading
        db = dataset_map[key][0](**kwargs)

        log.info("Dataset %s has %d images", key, len(db))
        db_list.append(db)

    return ConcatDataset(db_list)
