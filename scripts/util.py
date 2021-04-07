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

LOG = ttools.get_logger(__name__)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

class Params(object):
    def __init__(self, args):
        super(Params, self).__init__()

        self.working_dir = os.getcwd()
        self.output = args.output

        self.renderer = os.path.abspath(args.pbrt_exe)

'''
    Data:
        scene - scene to animate location
        spp - low_spp input
        gt_spp - amount of spp to render the target
        frames - amount of frames the animation is
    camera_translation - Vector indicating the translation of the camera
    camera_target - Vector indicating the translation of the target of the camera
    camera_up - Vector indicating the translation of the up-vector of the camera

    follow_target - 
    step - 
    roatation - 
        rot_axis -
        rot_angle -
'''
def animate_scene(data, camera_translation=np.zeros((3,)), camera_target=np.zeros((3,)),
        camera_up=np.zeros((3,)), follow_target=False, step=0.1, rotation=False, rot_axis=np.zeros((3,)), rot_angle=0):

    position = None
    look_at = None
    up = None

    roll = False

    scenes = [] # List of scenes that need to be rendered

    for curr_frame in range(data.frames):

        # Copy the scne to a temporary file
        tmp_scene = os.path.join(os.path.dirname(os.path.abspath(data.scene)), f'tmp_{data.scene_name}_{curr_frame}.pbrt')
        shutil.copy(data.scene, tmp_scene)

        # Update the camera position & write scene to temporary file to render
        try:
            for line in fileinput.input(tmp_scene, inplace = 1): 
                # For each line, chek if line contains the string
                if line.startswith('#'):
                    print(line, end='')
                elif "LookAt" in line:
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
                    # if current_frame == 15:
                    #     diff = (look_at - position)
                    #     axis = np.cross(diff, up)
                    #     rot_matrix = rotation_matrix(axis, -0.05)
                    #     look_at = np.dot(rot_matrix, look_at)
                    # elif current_frame == 30:
                    #     diff = (look_at - position)
                    #     axis = np.cross(diff, up)
                    #     rot_matrix = rotation_matrix(axis, 0.05)
                    #     look_at = np.dot(rot_matrix, look_at)

                    if follow_target:
                        diff = (look_at - position) * step
                        camera_translation = diff
                        camera_target = diff
                    
                    #Rotate the camera
                    if rotation:
                        rot_matrix = rotation_matrix(rot_axis, rot_angle)
                        look_at = np.dot(rot_matrix, look_at)

                    if roll:
                        axis = look_at
                        rot_matrix = rotation_matrix(axis, math.pi / 10)
                        up = np.dot(rot_matrix, up)

                    position += camera_translation
                    look_at += camera_target

                    coords[1:]= [str(x) for x in np.around(np.array(np.concatenate((position, look_at, up)), decimals=7))]

                    print(line.replace(line, " ".join(coords)))
                elif "Sampler" in line:
                    print(re.sub("\[\d*\]", f"[{data.gt_spp}]", line), end='')
                elif "Renderer" in line:
                    tmp = line.split(" ")
                    tmp[-1] = f"[{data.spp}]"
                    print(line.replace(line, " ".join(tmp)))
                else:
                    print(line, end='')
        except Exception as e:
            print(e)

        scenes.append(tmp_scene)

    return scenes