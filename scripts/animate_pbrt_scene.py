import argparse
from operator import pos
import ttools
import numpy as np
import subprocess
import shutil
import os
import glob
import fileinput
import re
import math

from multiprocessing import Pool, JoinableQueue, cpu_count
from generate_training_data import GeneratorParams, render
# from util import rotation_matrix, animate_scene
import util

LOG = ttools.get_logger(__name__)

class Params(object):

    def __init__(self, args):
        super(Params, self).__init__()

        self.working_dir = os.getcwd()
        self.output = args.output

        self.renderer = os.path.abspath(args.pbrt_exe)


def main(args): 
    ttools.set_logger(args.verbose)
    
    folder_mode = False
    if args.scenes != '':
        LOG.info("Selecting a random scene to be animated")
    
        # Get all pbrt files in scenes dir
        scenes = glob.glob(os.path.join(args.scenes, "*.pbrt"))
        folder_mode = True

    if args.width % args.tile_size != 0 or args.height % args.tile_size != 0:
        LOG.error("Block size should divide width and height.")
        raise ValueError("Block size should divide widt and height.")

    LOG.info("Starting job on worker %d of %d with %d threads" %
             (args.worker_id, args.num_workers, args.threads))   

    render_queue = JoinableQueue() 
    Pool(args.threads, render, (render_queue, ))

    for _ in range(args.count):

        scene_name = ''
        if folder_mode:
            random_scene = np.random.choice(scenes)
            LOG.info(f"Animated scene {random_scene.split('/')[-1]}")
            args.scene = random_scene
            scene_name = f"scene-{np.random.randint(2147000000)}_"

            # Add random animation
            follow = np.random.choice([True, False])
            follow = False

            if not follow:
                translation = np.random.uniform(0,2,3)
                look = np.random.uniform(0,2,3)
                animated_scenes = util.animate_scene(args, follow_target=follow, 
                    camera_translation=translation, camera_target=look)
            else:
                animated_scenes = util.animate_scene(args, follow_target=True)    
        else:
            animated_scenes = util.animate_scene(args, follow_target=True)

        for frame_idx, scene in enumerate(animated_scenes):
            dst_dir = os.path.abspath(os.path.join(args.output, f"{scene_name}frame-{frame_idx}"))
            os.makedirs(dst_dir, exist_ok=True)

            render_data = {"frame": frame_idx, "gen_params": Params(args), "scene_dir": dst_dir, 
                        "verbose": args.verbose, "clean": args.clean, "scene": scene}

            # Start rendering the scene (Frame by frame)
            render_queue.put(render_data, block=False)
            
            LOG.debug("Waiting for render queue.")
            render_queue.join()

            # Remove the temporary pbrt file
            os.remove(scene) 

def render(render_queue):
    while True:
        data = render_queue.get(block=True)
        frame = data["frame"]
        params = data["gen_params"]
        dst_dir = data["scene_dir"]
        verbose = data["verbose"]
        clean = data["clean"]
        LOG.info(f"Rendering frame {frame}")

        try:
            os.chdir(dst_dir)
            if verbose:
                stderr = None
            else:
                stderr = subprocess.DEVNULL
            ret = subprocess.check_output([params.renderer, data["scene"]],
                                          stderr=stderr)
            LOG.debug("Renderer output %s", ret)
        except Exception as e:
            LOG.warning("Rendering failed for scene %s: %s" % (dst_dir, e))
            shutil.rmtree(dst_dir)
            render_queue.task_done()
            continue
        os.chdir(params.working_dir)

        LOG.info("Finished rendering frame #%d", frame)
        render_queue.task_done()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #PBRT: "/pbrt/bin/pbrt"
    #OBJ2PBRT: "/pbrt/bin/obj.pbrt"
    #DATA: "/data/demo/scenegen_assets"

    # External binaries need to render the scene and convert the geometry
    parser.add_argument("pbrt_exe", help="path to the `pbrt` executable.")
    parser.add_argument("obj2pbrt_exe", help="path to PBRT's `obj2prt` "
                        "executable.")

    parser.add_argument('output')

    parser.add_argument('--scene', default='', help='Scene to be animated')
    parser.add_argument('--scenes', default='', help='Path to the folder containing scenes to be animated')

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
    parser.add_argument('--count', type=int, default=1,
                        help="number of scenes to generate per worker.")
    parser.add_argument('--frames', type=int, default=8,
                        help="number of frames of this scene to generate.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="number of scenes to generate before gathering"
                        " the outputs.")
    parser.add_argument("--verbose", dest="verbose", action="store_true",
                        default=False, help="Use verbose log messages.")

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