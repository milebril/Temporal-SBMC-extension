"""Train a model."""
import numpy as np
import torch as th
import time
import pyexr
import skimage.io as skio
from torch.utils.data import DataLoader
import os

from ttools.modules.image_operators import crop_like
from torch.optim import lr_scheduler

from sbmc import modules

import ttools

import sbmc
from sbmc import losses

LOG = ttools.get_logger(__name__)

def main(args):
    #Fix seed
    np.random.seed(0)
    th.manual_seed(0)

    data_args = dict(spp=args.spp, mode=sbmc.TilesDataset.KPCN_MODE if
                    args.kpcn_mode else sbmc.TilesDataset.SBMC_MODE,
                    load_coords=args.load_coords,
                    load_gbuffer=args.load_gbuffer, load_p=args.load_p,
                    load_ld=args.load_ld, load_bt=args.load_bt)
    
    # Make the checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Train with fixed spp
    data = sbmc.TilesDataset(args.data, **data_args)
    LOG.info("Training with a single sample count: %dspp" % args.spp)   
    
    if args.emil_mode:
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
        shuffle=False) 

    meta = dict(model_params=model_params, kpcn_mode=args.kpcn_mode,
            data_params=data_args)
        
    LOG.info("Model configuration: {}".format(model_params))

    # Enable CUDA//CPU
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # Loss functions
    loss_fn = losses.TonemappedRelativeMSE()
    rmse_fn = losses.RelativeMSE()

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if device == 'cuda':
        LOG.info('Using CUDA')
        loss_fn.cuda()
        rmse_fn.cuda()
        model.cuda()

    # Training params
    num_epochs = 10

    # Save randomly initialized model to compare with later epochs
    save_checkpoint(model, optimizer, os.path.join(args.checkpoint_dir, "start.pth"), -1)

    for epoch in range(num_epochs):
        # Start of an epoch
        for batch_idx, batch in enumerate(dataloader):
            # Start of a batch
            # Forward pass
            for k in batch:
                if not batch[k].__class__ == th.Tensor:
                    continue
                batch[k] = batch[k].to(device)
            output = model(batch)["radiance"]
            
            # Backward pass
            optimizer.zero_grad()
            target = crop_like(batch["target_image"], output)

            loss = loss_fn(output, target)
            # print(f'Epoch: {epoch}, batch: {batch_idx}, loss: {loss.item()}')
            loss.backward()

            # for i in range(len(list(model.parameters()))):
            #     t = list(model.parameters())[i].grad.device
            #     print(t)
                # if 'cpu' in str(t):
                #     print(list(model.parameters())[i].grad)

            # Clip the gradiants
            clip = 1000
            actual = th.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if actual > clip:
                LOG.info("Clipped gradients {} -> {}".format(clip, actual))

            optimizer.step()

            if (batch_idx == 0):
                rad = output.detach()
                save_img(rad, args.checkpoint_dir, str(epoch))

            printProgressBar(batch_idx+1, len(dataloader), prefix=f'Epoch {epoch}', suffix=f'{batch_idx+1}/{len(dataloader)} loss: {round(loss.item(), 3)}') # Print out progress after batch is finished    

        # End of an epoch
        scheduler.step()

    save_checkpoint(model, optimizer, os.path.join(args.checkpoint_dir, "training_end.pth"), num_epochs)

    # Check if training succeeded
    # tmp_model = sbmc.Multisteps(data.num_features, data.num_global_features,
    #                             ksize=args.ksize, splat=not args.gather,
    #                             pixel=args.pixel)
    # tmp_opt = th.optim.Adam(tmp_model.parameters(), lr=args.lr)
    # load_checkpoint(tmp_model, tmp_opt, os.path.join(args.checkpoint_dir, "start.pth"))
    # tmp_model.cuda()
    # compare_models(model, tmp_model)

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

def save_img(radiance, checkpoint_dir, name):
    data = th.clamp(radiance, 0)
    data /= 1 + data
    data = th.pow(data, 1.0/2.2)
    data = th.clamp(data, 0, 1)

    data = data[0, ...].cpu().detach().numpy().transpose([1, 2, 0])

    os.makedirs(checkpoint_dir, exist_ok=True)
    outputfile = os.path.join(checkpoint_dir, f'{name}.png')
    pyexr.write(outputfile, data)
    
    png = outputfile.replace(".exr", ".png")
    skio.imsave(png, (np.clip(data, 0, 1)*255).astype(np.uint8))


def save_checkpoint(model, optimizer, save_path, epoch):
    th.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

def load_checkpoint(model, optimizer, load_path):
    checkpoint = th.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if th.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

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