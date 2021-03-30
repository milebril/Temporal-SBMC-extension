import argparse
from operator import pos
import ttools
import numpy as np
import subprocess
import shutil
import os
import fileinput
import re
import math

from multiprocessing import Pool, JoinableQueue, cpu_count
from generate_training_data import GeneratorParams, render
from util import rotation_matrix

LOG = ttools.get_logger(__name__)

class Params(object):

    def __init__(self, args):
        super(Params, self).__init__()

        self.working_dir = os.getcwd()
        self.output = args.output

        self.renderer = os.path.abspath(args.pbrt_exe)


def main(args): 
    ttools.set_logger(args.verbose)

    if args.scene is None:
        LOG.error("Input scene has to be provided")

    if args.width % args.tile_size != 0 or args.height % args.tile_size != 0:
        LOG.error("Block size should divide width and height.")
        raise ValueError("Block size should divide widt and height.")

    LOG.info("Starting job on worker %d of %d with %d threads" %
             (args.worker_id, args.num_workers, args.threads))   

    # Define de desired camera movement in the animation
    camera_translation = np.array([0.0, 0.0, 0.0])
    camera_target = np.array([0.0, 0.0, 0.0])
    camera_up = np.array([0.0, 0.0, 0.0])

    position = None
    look_at = None
    up = None

    translate_look_dir = True

    # Roll camera
    roll_camera = False
    theta = 2 * math.pi / args.frames

    render_queue = JoinableQueue() 
    Pool(args.threads, render, (render_queue, ))

    current_frame = 0
    while True:
        if (current_frame >= args.frames):
            break

        # Copy the scne to a temporary file
        tmp_scene = os.path.join(os.path.dirname(os.path.abspath(args.scene)), 'tmp.pbrt')
        shutil.copy(args.scene, tmp_scene)
        change = ""
        # Update the camera position & write scene to temporary file
        try:
            for line in fileinput.input(tmp_scene, inplace = 1): 
                # For each line, chek if line contains the string
                if "LookAt" in line:
                    line_tmp = line.replace("  "," ")
                    coords = line_tmp.split(" ")
                    
                    # First iteration only
                    if position is None:
                        numbers = [float(x) for x in coords[1:9]]
                        numbers.append(float(coords[-1].split('/')[0]))
                        numbers = np.array(numbers)

                        position = numbers[0:3]
                        look_at = numbers[3:6]
                        up = numbers[6:]

                    # Look up
                    if current_frame == 15:
                        diff = (look_at - position)
                        axis = np.cross(diff, up)
                        rot_matrix = rotation_matrix(axis, -0.05)
                        look_at = np.dot(rot_matrix, look_at)
                    elif current_frame == 30:
                        diff = (look_at - position)
                        axis = np.cross(diff, up)
                        rot_matrix = rotation_matrix(axis, 0.05)
                        look_at = np.dot(rot_matrix, look_at)


                    if translate_look_dir:
                        diff = (look_at - position) * 0.1
                        camera_translation = diff
                        camera_target = diff
                    
                    #Rotate the camera
                    if roll_camera:
                        axis = look_at
                        rot_matrix = rotation_matrix(axis, theta)
                        up = np.dot(rot_matrix, up)

                    position += camera_translation
                    look_at += camera_target

                    # if current_frame == 10:
                    #     axis = np.cross(numbers[3:6], numbers[6:])
                    #     rot_matrix = rotation_matrix(axis, 0.1)
                    #     numbers[3:6] = np.dot(rot_matrix, numbers[3:6])

                    # if current_frame == 15:    
                    #     axis = np.cross(np.array(numbers[3:6]), np.array(numbers[6:]))
                    #     rot_matrix = rotation_matrix(axis, -0.1)
                    #     numbers[3:6] = np.dot(rot_matrix, numbers[3:6])

                    coords[1:]= [str(x) for x in np.array(np.concatenate((position, look_at, up)))]
                    print(line.replace(line, " ".join(coords)))
                elif "Sampler" in line:
                    print(re.sub("\[\d*\]", f"[{args.gt_spp}]", line), end='')
                elif "Renderer" in line:
                    tmp = line.split(" ")
                    tmp[-1] = f"[{args.spp}]"
                    print(line.replace(line, " ".join(tmp)))
                else:
                    print(line, end='')
        except Exception as e:
            print(e)
        dst_dir = os.path.abspath(os.path.join(args.output, f"frame-{current_frame}"))
        os.makedirs(dst_dir, exist_ok=True)

        render_data = {"frame": current_frame, "gen_params": Params(args), "scene_dir": dst_dir, 
                    "verbose": args.verbose, "clean": args.clean, "scene": tmp_scene}

        # Start rendering the scene (Frame by frame)
        render_queue.put(render_data, block=False)
        
        LOG.debug("Waiting for render queue.")
        render_queue.join()

        # Remove the temporary pbrt file
        # os.remove(tmp_scene) 
        
        current_frame += 1

def render(render_queue):
    while True:
        data = render_queue.get(block=True)
        frame = data["frame"]
        params = data["gen_params"]
        dst_dir = data["scene_dir"]
        verbose = data["verbose"]
        clean = data["clean"]
        LOG.info(f"Rendering scene '%s', frame {frame}", data["scene"].split('/')[-1])

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
    parser.add_argument('--scene', help='Scene to be animated')

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