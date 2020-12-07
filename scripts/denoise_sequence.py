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

"""
python scripts/denoise.py --input output/emil/training_sequence/render_samples_seq/scene-0_frame-0  --output output/emil/dataviz_sequence/denoised.exr --spp 4 --checkpoint data/pretrained_models/bako2017_finetuned/

"""

LOG = ttools.get_logger(__name__)

def main(args):
    data_root = os.path.abspath(args.input)
    name = os.path.basename(data_root)

    data_dirs = [f.path for f in os.scandir(data_root) if f.is_dir()]

    base_output = args.output

    for d in data_dirs:
        name = os.path.basename(d)
        print(d)
        args.input = d
        args.output = os.path.join(base_output, f"denoised-{name}.png")
        # Process(target=denoise, args=(args, )).start()
        denoise(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help="folder containing the sample .bin files.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="folder containing the model checkpoint.")
    parser.add_argument('--output', type=str, required=True,
                        help="output destination.")
    parser.add_argument('--spp', type=int,
                        help="number of samples to use as input.")
    parser.add_argument('--threads', type=int, default=1,
                        help="number of threads to use for denoising scenes.")
    parser.add_argument("--tile_size", default=1024, help="We process in tiles"
                        " to limit GPU memory usage. This is the tile size.")
    parser.add_argument("--tile_pad", default=256, help="We process in tiles"
                        " to limit GPU memory usage. This is the padding"
                        " around tiles, for overlapping tiles.")
    args = parser.parse_args()
    ttools.set_logger(True)
    main(args)