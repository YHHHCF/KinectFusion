# measurement.py implements surface measurement component
# This component is responsible to generate vertex and normal
# maps from a given depth map frame

import open3d as o3d
import numpy as np
import cv2
import sys
sys.path.append('../data/')
from dataloader import *
from camera import *
import matplotlib.pyplot as plt

# generate vertex map, normal map and valid mask from depth map
def vertex_normal_from_depth(depth_numpy):
    h, w = depth_numpy.shape
    print(h, w)


if __name__ == "__main__":
    depth_path = "../data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png"
    depth_numpy = load_depth_as_numpy(depth_path)
    print(depth_numpy.shape)

    plt.imshow(depth_numpy)
    plt.show()

    vertex_normal_from_depth(depth_numpy)
