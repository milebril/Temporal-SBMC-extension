import numpy as np
import torch as th
import cv2
import argparse
import tempfile
from torch.utils.data import DataLoader
import os
import pyexr
import cv2
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
    # model_one = sbmc.Multisteps(data.num_features, data.num_global_features)
    model_one = sbmc.RecurrentMultisteps(data.num_features, data.num_global_features)
    model_one.load_state_dict(temp['model'])
    model_one.train(False)

    temp = th.load(f"{args.model2}", map_location=th.device('cpu'))
    model_two = sbmc.Multisteps(data.num_features, data.num_global_features)
    model_two.load_state_dict(temp['model'])
    model_two.train(False)

    device = "cuda" if th.cuda.is_available() else "cpu"
    device = "cpu"
    if (device == "cuda"):
        LOG.info("Using CUDA")
        model_one.cuda()
        model_two.cuda()

    rmse_checker = losses.RelativeMSE()
    rmse_checker.to(device)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.amount:
            break

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
    #     save_compare_frame(output1, output2, tgt)
    # make_compare_video(args.save_dir)

frames = []
def save_compare_frame(radiance1, radiance2, tgt):
    # Difference between models and ground thruth
    diff_model1 = (radiance1 - tgt).abs()
    diff_model2 = (radiance2 - tgt).abs()

    first_row = th.cat([radiance1,    diff_model1], -1)
    second_row  = th.cat([radiance2,    diff_model2], -1)

    data = th.cat([first_row, second_row], -2)

    data = th.clamp(data, 0)
    data /= 1 + data
    data = th.pow(data, 1.0/2.2)
    data = th.clamp(data, 0, 1)

    data = data[0, ...].cpu().detach().numpy().transpose([1, 2, 0])

    # Clip to 0-255 to remove HDR and pure radiance estimates + change to BGR color spectrum for opencv
    frames.append(cv2.cvtColor((np.clip(data, 0, 1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

def make_compare_video(location):
    height, width, layers = frames[0].shape

    # Write to video
    out = cv2.VideoWriter(f'{location}/compare_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
    
    # Stitch 5 times to create loop
    for _ in range(10):
        for i in range(len(frames)):
            out.write(frames[i])
        frames.reverse()

    out.release()

def save_img(radiance1, radiance2, low_radiance, tgt, checkpoint_dir, name):
    tmp_empty = th.zeros_like(radiance1) # Empty filler tensor

    # Difference between models and ground thruth
    diff_model1 = (radiance1 - tgt).abs()
    diff_model2 = (radiance2 - tgt).abs()

    # Create output data in the form:
    #   low spp input -- 
    #   ouput model1  -- Diff with tgt
    #   ouput model2  -- Diff with tgt
    #   tgt           -- 
    first_row  = th.cat([tmp_empty, low_radiance, tmp_empty], -1)
    second_row = th.cat([tmp_empty, radiance1,    diff_model1], -1)
    third_row  = th.cat([tmp_empty, radiance2,    diff_model2], -1)
    fourth_row = th.cat([tmp_empty, tgt,          tmp_empty], -1)

    # Concate the data in a vertical stack
    data = th.cat([first_row, second_row, third_row, fourth_row], -2)

    data = th.clamp(data, 0)
    data /= 1 + data
    data = th.pow(data, 1.0/2.2)
    data = th.clamp(data, 0, 1)

    data = data[0, ...].cpu().detach().numpy().transpose([1, 2, 0])
    data = np.ascontiguousarray(data)

    # Add text to the images
    jump = radiance1.size()[2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(data, '4spp', (10, jump * 0 + 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(data, 'Model 1', (10, jump * 1 + 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(data, 'Model 2', (10, jump * 2 + 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(data, 'Target', (10, jump * 3 + 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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
    parser.add_argument(
        '--amount', required=False, type=int,default=1, help="Amount of frames to denoise and compare")

    parser.add_argument('--spp', type=int,
                    help="number of samples to use as input.")

    args = parser.parse_args()
    ttools.set_logger(True)
    main(args)