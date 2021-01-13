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
"""Train a model."""
import numpy as np
import torch as th
import os
import pyexr

from torch.utils.data import DataLoader

from sbmc import modules

import ttools

import sbmc


LOG = ttools.get_logger(__name__)

def train(dataloader, num_epochs, interface, val_dataloader=None):
    
    epoch = 0
    while num_epochs is None or epoch < num_epochs:
        LOG.info(f"Started epoch {epoch}")
        for batch_idx, batch in enumerate(dataloader):
            # Start a batch
            fwd_result, hidden_result = interface.forward(batch) # One Pass through the model with a batch
            bwd_result = interface.backward(batch, fwd_result) # Backward pass of the result
        LOG.info(f"Ended epoch {epoch}")

        # Validate
        # if val_dataloader:
        #     with th.no_grad():
        #         val_data = self.__validation_start(
        #             val_dataloader)  # data interface adapter
        #         for batch_idx, batch in enumerate(val_dataloader):
        #             if not self._keep_running:
        #                 self._stop()
        #                 return
        #             fwd_result = self.__forward_step(batch)
        #             val_data = self.__validation_update(
        #                 batch, fwd_result, val_data)
        #         self.__validation_end(val_data)

        epoch += 1

    stop()

def stop():
    pass


def main(args):
    # Fix seed
    np.random.seed(0)
    th.manual_seed(0)

    # Parameterization of the dataset (shared between train/val)
    data_args = dict(spp=args.spp, mode=sbmc.TilesDataset.KPCN_MODE if
                     args.kpcn_mode else sbmc.TilesDataset.SBMC_MODE,
                     load_coords=args.load_coords,
                     load_gbuffer=args.load_gbuffer, load_p=args.load_p,
                     load_ld=args.load_ld, load_bt=args.load_bt)

    if args.randomize_spp:
        if args.bs != 1:
            LOG.error("Training with randomized spp is only valid for"
                      "batch_size=1, got %d", args.bs)
            raise RuntimeError("Incorrect batch size")
        data = sbmc.MultiSampleCountDataset(
            args.data, **data_args)
        LOG.info("Training with randomized sample count in [%d, %d]" % (
            2, args.spp))
    else:
        data = sbmc.TilesDataset(args.data, **data_args)
        LOG.info("Training with a single sample count: %dspp" % args.spp)

    if args.kpcn_mode:
        LOG.info("Model: pixel-based comparison from [Bako2017]")
        model = sbmc.KPCN(data.num_features, ksize=args.ksize)
        model_params = dict(ksize=args.ksize)
    elif args.emil_mode:
        LOG.info("Model: Temporal Sample-Based Denoising [Peters2021]")
        model = sbmc.RecurrentMultisteps(data.num_features, data.num_global_features,
                                ksize=args.ksize, splat=not args.gather,
                                pixel=args.pixel)
        model_params = dict(ksize=args.ksize, gather=args.gather,
                            pixel=args.pixel)
    else:
        LOG.info("Model: sample-based [Gharbi2019]")
        model = sbmc.Multisteps(data.num_features, data.num_global_features,
                                ksize=args.ksize, splat=not args.gather,
                                pixel=args.pixel)
        model_params = dict(ksize=args.ksize, gather=args.gather,
                            pixel=args.pixel)

    dataloader = DataLoader(
        data, batch_size=args.bs, num_workers=args.num_worker_threads,
        shuffle=False) # Don't shuffle as we want to learn to denoise sequences


    # Validation set uses a constant spp
    val_dataloader = None
    if args.val_data is not None:
        LOG.info("Validation set with %dspp" % args.spp)
        val_data = sbmc.TilesDataset(args.val_data, **data_args)
        val_dataloader = DataLoader(
            val_data, batch_size=args.bs, num_workers=1, shuffle=False)
    else:
        LOG.info("No validation set provided")

    meta = dict(model_params=model_params, kpcn_mode=args.kpcn_mode,
                data_params=data_args)

    LOG.info("Model configuration: {}".format(model_params))

    # Load latest model
    model_location =  os.path.join(args.checkpoint_dir, "training_end.pth")
    if os.path.isfile(model_location):
        temp = th.load(model_location, map_location=th.device('cpu'))
        model.load_state_dict(temp['model'])
    else:
        LOG.info("Loading SBMC weights into Temporal model")
        gharbi = "/home/emil/Documents/Temporal-SBMC-extension/data/pretrained_models/gharbi2019_sbmc/final.pth"
        pre_trained_model = th.load(gharbi, map_location=th.device('cpu'))
        new = list(pre_trained_model['model'].items())
        my_model_kvpair = model.state_dict()

        count=0
        for key,value in my_model_kvpair.items():
            layer_name, weights = new[count]

            # Skip the modules with recurrent connections   
            if 'propagation_02' in layer_name and 'left' in layer_name:
                count+=1
                continue
            # print(f"Layer: {layer_name}")
            my_model_kvpair[key] = weights
            count+=1

        model.load_state_dict(my_model_kvpair)

    # Lock all other parameters
    # for name, layer in model.named_modules():
    #     # Skip recurrent layers
    #     if isinstance(layer, modules.RecurrentConvChain):
    #         continue
    #     for param in layer.parameters():
    #         param.requires_grad = False 

    # Set up the interface
    interface = sbmc.SampleBasedDenoiserInterface(
        model, lr=args.lr, cuda=False)

    num_epochs = 2
    epoch = 0

    while num_epochs is None or epoch < num_epochs:
        nbatches = len(dataloader)

        # Reset hidden-state at the start of an epoch
        hidden_state = {
            0: None,
            1: None
        }

        for batch_idx, batch in enumerate(dataloader):
            # Start a batch
            fwd_result = interface.forward(batch, hidden=hidden_state) # One Pass through the model with a batch
            hidden_state = fwd_result["hidden"]
            hidden_state[0].detach_()
            hidden_state[1].detach_()
            bwd_result = interface.backward(batch, fwd_result) # Backward pass of the result
            printProgressBar(batch_idx+1, nbatches, prefix=f'Epoch {epoch}', suffix=f'{batch_idx+1}/{nbatches} loss: {round(bwd_result["loss"], 3)} RMSE: {round(bwd_result["rmse"],3)}') # Print out progress after batch is finished

        # Save model-state after an epoch
        # LOG.debug(f"Saving epoch {epoch} to disk")
        save(args.checkpoint_dir, f'epoch_{epoch}.pth', model, meta)
        
        # Save a denoised image per epoch
        out = fwd_result['radiance']

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        outputfile = os.path.join(args.checkpoint_dir, f'epoch_{epoch}')
        pyexr.write(outputfile, out)
        
        png = outputfile.replace(".exr", ".png")
        skio.imsave(png, (np.clip(out, 0, 1)*255).astype(np.uint8))

        # Validate
        # if val_dataloader:
        #     with th.no_grad():
        #         val_data = self.__validation_start(
        #             val_dataloader)  # data interface adapter
        #         for batch_idx, batch in enumerate(val_dataloader):
        #             if not self._keep_running:
        #                 self._stop()
        #                 return
        #             fwd_result = self.__forward_step(batch)
        #             val_data = self.__validation_update(
        #                 batch, fwd_result, val_data)
        #         self.__validation_end(val_data)

        epoch += 1
    
    # Save the final model state
    save(args.checkpoint_dir, f'training_end.pth', model, meta)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def save(root, filename, model, meta, optimizers=None, extras=None):
        """Save model, metaparams and extras to relative path.

        Args:
          path (string): relative path to the file being saved (without extension).
          extras (dict): extra user-provided information to be saved with the model.
        """

        model_state = model.state_dict()

        opt_dicts = []
        if optimizers is not None:
            for opt in optimizers:
                opt_dicts.append(opt.state_dict())

        os.makedirs(root, exist_ok=True)
        filename = os.path.join(root, filename)
        th.save({'model': model_state,
                 'meta': meta,
                 'extras': extras,
                 'optimizers': opt_dicts,
                 }, filename)
        LOG.debug("Checkpoint saved to \"{}\"".format(filename))

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

if __name__ == "__main__":
    parser = ttools.BasicArgumentParser()
    parser.add_argument(
        '--spp', type=int, default=8, help="Max number of samples per pixel.")

    # Model parameters
    parser.add_argument(
        '--kpcn_mode', dest="kpcn_mode", action="store_true", default=False,
        help="if True, use the model from [Bako2017]: useful for comparison.")
    parser.add_argument(
        '--emil_mode', dest="emil_mode", action="store_true", default=False,
        help="if True, use the model from [Peters2021]: temporal extension to [Gharbi2019].")
    parser.add_argument(
        '--gather', dest="gather", action="store_true", default=False,
        help="if True, use gather kernels instead of splat.")
    parser.add_argument(
        '--pixel', dest="pixel", action="store_true", default=False,
        help="if True, use per-pixel model instead of samples.")
    parser.add_argument(
        '--ksize', type=int, default=21, help="Size of the kernels")

    # Data configuration
    parser.add_argument('--constant_spp', dest="randomize_spp",
                        action="store_false", default=True)

    parser.add_argument('--dont_use_coords', dest="load_coords",
                        action="store_false", default=True)
    parser.add_argument('--dont_use_gbuffer', dest="load_gbuffer",
                        action="store_false", default=True)
    parser.add_argument('--dont_use_p', dest="load_p",
                        action="store_false", default=True)
    parser.add_argument('--dont_use_ld', dest="load_ld",
                        action="store_false", default=True)
    parser.add_argument('--dont_use_bt', dest="load_bt",
                        action="store_false", default=True)

    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)
