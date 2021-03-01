import numpy as np
import torch as th
import argparse
import tempfile
from torch.utils.data import DataLoader
import os
import pyexr
import skimage.io as skio
from ttools.modules.image_operators import crop_like

from sbmc import losses
from sbmc import modules

import ttools

import sbmc

LOG = ttools.get_logger(__name__)

#'ksize': 21, 'gather': False, 'pixel': False

def main(args):
    if not os.path.exists(args.data):
        raise ValueError("input {} does not exist".format(args.data))

    # Load the data
    data_params = dict(spp=args.spp)

    data = sbmc.FullImagesDataset(args.data, **data_params)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    # Load the two models
    temp = th.load(f"{args.model1}", map_location=th.device('cpu'))
    model_one = sbmc.Multisteps(data.num_features, data.num_global_features)
    model_one.load_state_dict(temp['model'])
    model_one.train(False)

    temp = th.load(f"{args.model2}", map_location=th.device('cpu'))
    model_two = sbmc.Multisteps(data.num_features, data.num_global_features)
    model_two.load_state_dict(temp['model'])
    model_two.train(False)

    device = "cuda" if th.cuda.is_available() else "cpu"
    if (device == "cuda"):
        LOG.info("Using CUDA")
        model_one.cuda()
        model_two.cuda()

    rmse_checker = losses.RelativeMSE()
    rmse_checker.to(device)

    for batch_idx, batch in enumerate(dataloader):
        for k in batch.keys():
            if not batch[k].__class__ == th.Tensor:
                continue
            batch[k] = batch[k].to(device) #Sets the tensors to the correct device type

        # Compute the radiances using the two models
        with th.no_grad():
            output1 = model_one(batch)["radiance"]
            output2 = model_two(batch)["radiance"]

        # Get the input image and ground thruth for comparison
        tgt = crop_like(batch["target_image"], output1)
        low_spp = crop_like(batch["low_spp"], output1)

        # Compare to ground thruth
        with th.no_grad():
            rmse1 = rmse_checker(output1, tgt)
            rmse2 = rmse_checker(output2, tgt)
        
        LOG.info(f"Model 1 denoised with rmse: {rmse1} || Model 2 denoised with rmse: {rmse2}")
        if rmse2 < rmse1:
            LOG.info("Model 2 outperformed model 1")
        else:
            LOG.info("Model 1 outperformed model 2")

        save_img(output1, output2, low_spp, tgt, args.save_dir, str(batch_idx))

def save_img(radiance1, radiance2, low_radiance, tgt, checkpoint_dir, name):
    data = th.cat([low_radiance, radiance1, radiance2, tgt], -2)

    data = th.clamp(data, 0)
    data /= 1 + data
    data = th.pow(data, 1.0/2.2)
    data = th.clamp(data, 0, 1)

    data = data[0, ...].cpu().detach().numpy().transpose([1, 2, 0])

    os.makedirs(checkpoint_dir, exist_ok=True)
    outputfile = os.path.join(checkpoint_dir, f'{name}.png')
    pyexr.write(outputfile, data)
    
    png = outputfile.replace(".exr", ".png")
    skio.imsave(png, (np.clip(data, 0, 1)*255).astype(np.uint8))

def load_model(model, load_path):
    checkpoint = th.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model1', required=True, help="path to the first model")
    parser.add_argument(
        '--model2', required=True, help="path to the second model")
    parser.add_argument(
        '--save_dir', required=True, help="path to the dir where everything has to be saved")
    parser.add_argument(
        '--data', required=True, help="path to the training data.")

    parser.add_argument('--spp', type=int,
                    help="number of samples to use as input.")

    args = parser.parse_args()
    ttools.set_logger(True)
    main(args)