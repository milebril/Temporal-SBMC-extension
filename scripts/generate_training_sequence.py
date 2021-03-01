import uuid
import time
import argparse
from multiprocessing import Pool, JoinableQueue, cpu_count
import os
import subprocess
import shutil

#repurpose
from generate_training_data import GeneratorParams, render

import numpy as np

import ttools

import sbmc.scene_generator as scenegen

LOG = ttools.get_logger(__name__)

"""
    Create 1 random scene, and fly with the camere through it
"""

def _validate_render(path):
    """
    Remove all intermediate files and directories used to generate the .bin
    data.

    Args:
        path(str): path to the output directory to clean.
    """
    files = os.listdir(path)
    exts = [os.path.splitext(f)[-1] for f in files]
    exts = set(exts)
    if ".bin" not in exts:
        return False
    return True

def _clean_bin_folder(path):
    """
    Remove all intermediate files and directories used to generate the .bin
    data.

    Args:
        path(str): path to the output directory to clean.
    """
    files = os.listdir(path)
    for file in files:
        file = os.path.join(path, file)
        if os.path.splitext(file)[-1] != ".bin":
            if os.path.islink(file):
                os.remove(file)
            elif os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)

def _random_dirname():
    """Generates a directory name for the random scene.

    Uses the host name, timestamp and a random UUID to disambiguate
    scens in a distributed rendering context.
    """
    hostname = "Emil"
    date = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
    name = "%s_%s_%s" % (hostname, date, str(uuid.uuid4())[:8])
    return name

def create_scene_file(q, render_queue):
    while True:
        data = q.get(block=True)

        idx = data["idx"]
        params = data["gen_params"]
        rparams = data["render_params"]

        curr_frame = 0

        LOG.debug("Creating scene {}, frame {}".format(idx, curr_frame))
        np.random.seed(data['random'])

        # Create container
        dirname = "render_samples_seq"
        dst_dir = os.path.abspath(os.path.join(params.output, dirname, f"scene-{idx}_frame-{0}"))

        try:
            LOG.debug("Setting up folder {}".format(dst_dir))
            os.makedirs(dst_dir, exist_ok=True)
            os.makedirs(os.path.join(dst_dir, f"geometry_scene#{idx}"), exist_ok=True)
        except Exception as e:
            LOG.warning(
                "Could not setup directories %s, " \
                "continuing to next scene: %s" % (dst_dir, e))
            q.task_done()
            continue
        LOG.debug("{} directory ready".format(dst_dir))

        # Randomize resolution
        # scale = np.random.choice([1, 2, 4, 8])
        # width = rparams["width"]*scale
        # height = rparams["height"]*scale

        # Maintain the size constant despite the resolution change
        # rparams["random_crop_w"] = rparams["width"]
        # rparams["random_crop_h"] = rparams["height"]
        # rparams["width"] = width
        # rparams["height"] = height

        # parameters = {"spp": rparams.spp, "gt_spp": rparams.gt_spp, "width":
        #               width, "height": height, "path_depth":
        #               rparams.path_depth, "random_crop_x": rparams.width,
        #               "random_crop_h": rparams.height, "tile_size":
        #               rparams.tile_size}
        renderer = scenegen.Renderer(**rparams)

        scn = scenegen.Scene(renderer=renderer)

        max_attempts = 20
        attempt = 0
        try:
            gen = np.random.choice(params.gen)
            # while not gen.sample_sequence(scn, dst_dir, idx=idx):
            # while not gen.sample_cornellbox_scene(scn, dst_dir, idx=idx):
            while not gen.sample_wall_Scene(scn, dst_dir, idx=idx):
                attempt += 1
                LOG.warning("Sampling another Scene {}".format(gen))
                if attempt == max_attempts:
                    break

            if attempt == max_attempts:
                LOG.warning(
                    "Could not generate a scene, continuing to next seed")
                q.task_done()
                continue
        except Exception as e:
            LOG.warning(
                "Scene sampling failed at attempt {}: {}, continuing to next"
                " scene".format(attempt, e))
            q.task_done()
            continue
        
        # Render the frames
        for i in range(params.frames):
            dst_dir = os.path.abspath(os.path.join(params.output, "render_samples_seq" ,f"scene-{idx}_frame-{i}"))
            try:
                os.makedirs(dst_dir, exist_ok=True)
            except Exception as e:
                LOG.warning(
                    "Could not setup directories %s, " \
                    "continuing to next scene: %s" % (dst_dir, e))
                q.task_done()
                continue

            try:
                scn_file = os.path.join(dst_dir, "scene.pbrt")
                with open(scn_file, 'w') as fid:
                    fid.write(scn.pbrt())
            except:
                LOG.error("Failed to save .pbrt file, continuing")
                q.task_done()
                continue


            render_data = {"idx": idx, "gen_params": params, "render_params":
                        rparams, "scene_dir": dst_dir, "verbose":
                        data["verbose"], "clean": data["clean"]}
            LOG.info("Adding scene #%d f%d to the render queue", idx, i)
            render_queue.put(render_data, block=False)

            # Move camera in scene
            scn.translate_camera([0.02, 0, 0])

        q.task_done()
        continue

def main(args):
    ttools.set_logger(args.verbose)

    if args.width % args.tile_size != 0 or args.height % args.tile_size != 0:
        LOG.error("Block size should divide width and height.")
        raise ValueError("Block size should divide widt and height.")

    LOG.info("Starting job on worker %d of %d with %d threads" %
             (args.worker_id, args.num_workers, args.threads))
    
    gen_params = GeneratorParams(args)
    render_params = dict(spp=args.spp, gt_spp=args.gt_spp, height=args.height,
                         width=args.width, path_depth=args.path_depth,
                         tile_size=args.tile_size)

    scene_queue = JoinableQueue() # Scene creation
    render_queue = JoinableQueue() # Rendering queue
    Pool(args.threads, create_scene_file,
                      (scene_queue, render_queue))
    Pool(args.threads, render, (render_queue, ))

    LOG.info("Generating %d random scenes", args.count)

    count = 0  # count the number of scenes generated
    while True:
        # Generate a batch of scene files (to limit memory usage, we do not
        # queue all scenes at once.
        for _ in range(min(args.batch_size, args.count)):
            idx = args.start_index + count*args.num_workers + args.worker_id
            data = {
                "idx": idx,
                "gen_params": gen_params,
                "render_params": render_params,
                "verbose": args.verbose,
                "clean": args.clean,
                "random": np.random.randint(0, 2147000000)
            }
            if args.count > 0 and count == args.count:
                break
            scene_queue.put(data, block=False)
            count += 1

        LOG.debug("Waiting for scene queue.")
        scene_queue.join()

        LOG.debug("Waiting for render queue.")
        render_queue.join()

        LOG.debug("Finished all queues.")

        # Only render up to `args.count` images
        if args.count > 0 and count == args.count:
            break

    LOG.debug("Shutting down the scene generator")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #PBRT: "/pbrt/bin/pbrt"
    #OBJ2PBRT: "/pbrt/bin/obj.pbrt"
    #DATA: "/data/demo/scenegen_assets"

    # External binaries need to render the scene and convert the geometry
    parser.add_argument("pbrt_exe", help="path to the `pbrt` executable.")
    parser.add_argument("obj2pbrt_exe", help="path to PBRT's `obj2prt` "
                        "executable.")

    # Data and output folders
    parser.add_argument('assets', help="path to the assets to use.")
    parser.add_argument('output')

    # parser.add_argument('--suncg_root', type=str, default="local_data/suncg")

    # Distributed workers params
    parser.add_argument('--start_index', type=int, default=0,
                        help="index of the first scene to generate.")
    parser.add_argument('--worker_id', type=int, default=0,
                        help="id of the current worker.")
    parser.add_argument('--num_workers', type=int, default=1,
                        help="number of distributed workers in the swarm.")
    parser.add_argument('--threads', type=int,
                        default=max(cpu_count() // 2, 1),
                        help="threads to use for parallelized work.")
    parser.add_argument('--count', type=int, default=-1,
                        help="number of scenes to generate per worker.")
    parser.add_argument('--frames', type=int, default=8,
                        help="number of frames of this scene to generate.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="number of scenes to generate before gathering"
                        " the outputs.")
    parser.add_argument("--verbose", dest="verbose", action="store_true",
                        default=False, help="Use verbose log messages.")

    # Generators configuration
    parser.add_argument('--generators', nargs="+",
                        default=["OutdoorSceneGenerator"],
                        choices=scenegen.generators.__all__,
                        help="scene generator class to use.")

    # Rendering parameters
    parser.add_argument('--spp', type=int, default=32)
    parser.add_argument('--gt_spp', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--path_depth', type=int, default=5)
    parser.add_argument('--tile_size', type=int, default=128)

    parser.add_argument('--no-clean', dest="clean", action="store_false",
                        default=True)

    main(parser.parse_args())