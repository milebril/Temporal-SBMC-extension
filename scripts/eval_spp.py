#!/usr/bin/env python
# encoding: utf-8
# Sample-based Monte Carlo Denoising using a Kernel-Splatting Network
# Michaël Gharbi Tzu-Mao Li Miika Aittala Jaakko Lehtinen Frédo Durand
# Siggraph 2019
#
# Copyright (c) 2019 Michaël Gharbi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Denoise an image using a previously trained model."""

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
from sbmc import losses

import ttools
from ttools.modules.image_operators import crop_like

import sbmc


LOG = ttools.get_logger(__name__)

def main(args):
    if not os.path.exists(args.data):
        raise ValueError("input {} does not exist".format(args.data))

    # Load the data
    data_params = dict(spp=args.spp)

    data = sbmc.FullImagesDataset(args.data, **data_params)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    # Load the model
    temp = th.load(f"{args.model1}", map_location=th.device('cpu'))
    # model_one = sbmc.Multisteps(data.num_features, data.num_global_features)
    model_one = sbmc.RecurrentMultisteps(data.num_features, data.num_global_features)
    model_one.load_state_dict(temp['model'])
    model_one.train(False)

    device = "cuda" if th.cuda.is_available() else "cpu"
    if (device == "cuda"):
        LOG.info("Using CUDA")
        model_one.cuda()

    rmse_checker = losses.RelativeMSE()
    rmse_checker.to(device)

    start = 0

    spps = [1, 4, 16, 32, 64]
    radiances = []

    for spp in spps:
        data_params["spp"] = spp
        data = sbmc.FullImagesDataset(args.data, **data_params)
        dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

        batch = next(iter(dataloader))

        for k in batch.keys():
            if not batch[k].__class__ == th.Tensor:
                continue
            batch[k] = batch[k].to(device) # Sets the tensors to the correct device type

        # Compute the radiances using the two models
        with th.no_grad():
            output = model_one(batch)["radiance"]
        
        tgt = crop_like(batch["target_image"], output)

        radiances.append(output)
    
    save_img(radiances, tgt, args.save_dir)

def save_img(radiances, tgt, checkpoint_dir):
    tmp_empty = th.zeros_like(radiances[0]) # Empty filler tensor

    # Difference between models and ground thruth
    # diff_model1 = (radiance1 - tgt).abs()
    # diff_model2 = (radiance2 - tgt).abs()

    # Create output data in the form:
    #   low spp input -- 
    #   ouput model1  -- Diff with tgt
    #   ouput model2  -- Diff with tgt
    #   tgt           -- 
    # first_row  = th.cat([tmp_empty, low_radiance, tmp_empty], -1)
    # second_row = th.cat([tmp_empty, radiance1,    diff_model1], -1)
    # third_row  = th.cat([tmp_empty, radiance2,    diff_model2], -1)
    # fourth_row = th.cat([tmp_empty, tgt,          tmp_empty], -1)

    # Concate the data in a vertical stack
    # data = th.cat([first_row, second_row, third_row, fourth_row], -2)
    radiances.append(tgt)
    data = th.cat(radiances, -1)

    data = th.clamp(data, 0)
    data /= 1 + data
    data = th.pow(data, 1.0/2.2)
    data = th.clamp(data, 0, 1)

    data = data[0, ...].cpu().detach().numpy().transpose([1, 2, 0])
    data = np.ascontiguousarray(data)

    # Add text to the images

    os.makedirs(checkpoint_dir, exist_ok=True)
    outputfile = os.path.join(checkpoint_dir, f'spp.png')
    pyexr.write(outputfile, data)
    
    png = outputfile.replace(".exr", ".png")
    skio.imsave(png, (np.clip(data, 0, 1)*255).astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model1', required=True, help="path to the first model")
    parser.add_argument(
        '--save_dir', required=True, help="path to the dir where everything has to be saved")
    parser.add_argument(
        '--data', required=True, help="path to the training data.")
    parser.add_argument(
        '--amount', required=False, type=int,default=1, help="Amount of frames to denoise and compare")

    parser.add_argument('--spp', type=int,
                    help="number of samples to use as input.")

    args = parser.parse_args()
    ttools.set_logger(True)
    main(args)
