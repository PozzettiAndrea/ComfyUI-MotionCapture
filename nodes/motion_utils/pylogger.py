from time import time
import logging
import torch


def sync_time():
    torch.cuda.synchronize()
    return time()


Log = logging.getLogger()
Log.time = time
Log.sync_time = sync_time

# Set default
Log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatstring = "[%(asctime)s][%(levelname)s] %(message)s"
datefmt = "%m/%d %H:%M:%S"
ch.setFormatter(logging.Formatter(formatstring, datefmt=datefmt))

Log.addHandler(ch)
