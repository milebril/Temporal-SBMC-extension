import os
import argparse
import shutil
import tempfile
import time
import pyexr

import torch as th
import numpy as np
from torch.utils.data import DataLoader
import skimage.io as skio
from multiprocessing import Pool, JoinableQueue, cpu_count, Process

import ttools

from denoise import denoise

import sbmc

inputs = "/home/emil/Documents/sbmc/output/emil/training_sequence/render_samples_seq"
checkpoint = "/home/emil/Documents/sbmc/data/pretrained_models/gharbi2019_sbmc"

if not os.path.exists(inputs):
    raise ValueError("input {} does not exist".format(args.input))

data_root = os.path.abspath(inputs)
print("ROOT: ", data_root)

# Load everything into a tpm folder and link it up
name = os.path.basename(data_root)
tmpdir = tempfile.mkdtemp()
os.symlink(data_root, os.path.join(tmpdir, name))

# LOG.info("Loading model {}".format(checkpoint))
meta_params = ttools.Checkpointer.load_meta(checkpoint)

# LOG.info("Setting up dataloader")
data_params = meta_params["data_params"]
data_params["spp"] = 4

data = sbmc.FullImagesDataset(os.path.join(tmpdir, name), **data_params)


