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

debug = False

# generate vertex map, normal map in o3d's t space
# TODO: also generate a valid mask and think of filter out invalid points from ICP
# TODO: remove the filter out invalid depth logic in point_cloud_from_depth() after the above logic is done
# TODO: if needed, create a o3d point cloud to make sure vertex and normal maps format matches with prediction step for ICP
def maps_from_depth(depth_numpy, camera):
    vertex_map_numpy = camera.vertex_map_from_depth(depth_numpy).astype(np.float32)

    # create vertex map as o3d image
    vertex_map_o3d = o3d.t.geometry.Image(vertex_map_numpy)

    # create normal map
    normal_map_o3d = vertex_map_o3d.create_normal_map()

    if debug:
        print("vertex map min, max:", np.min(vertex_map_numpy), np.max(vertex_map_numpy))
        print("normal map min, max:", np.min(np.asarray(normal_map_o3d)), np.max(np.asarray(normal_map_o3d)))

        plt.imshow(np.asarray(vertex_map_o3d))
        plt.show()
        plt.imshow(np.asarray(normal_map_o3d))
        plt.show()

    return vertex_map_o3d, normal_map_o3d


if __name__ == "__main__":
    camera = Camera()

    depth_path = "../data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png"
    depth_numpy = load_depth_as_numpy(depth_path)
    print("depth size:", depth_numpy.shape)

    plt.imshow(depth_numpy)
    plt.show()

    maps_from_depth(depth_numpy, camera)
