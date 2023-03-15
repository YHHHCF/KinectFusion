# update.py implements TSDF update component
# This component takes a depth map frame and extrinsics to compute
# the local TSDF at global position and fuse it with global TSDF
# This implementation adopts Open3D implementation of TSDF, which is
# based on a HashMap (eg. with 10000 slots) of blocks where each block
# is a small cube (eg. 16x16x16) of of voxels. The Open3D is a high quality and
# high performance (O(surface_area) space complexity, O(1) time complexity) implementation.
# Ref: http://www.open3d.org/docs/latest/python_api/open3d.t.geometry.VoxelBlockGrid.html

import open3d as o3d
import open3d.core as o3c
import numpy as np
import os
import sys
sys.path.append('../data/')
from dataloader import *
from camera import *
from prediction import *

# TODO: play with (and understand) vbg lower level implementations, read C++ source code

# TODO: tune the parameters: voxel_size, block_resolution and block_count
def create_vbg():
    vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight'), # signed distance value, weight (number of observations)
            attr_dtypes=(o3c.float32, o3c.float32),
            attr_channels=((1), (1)),
            voxel_size=3.0 / 512, # 3 / 512 = 0.00586 m
            block_resolution=16, # 16x16x16 voxels per block
            block_count=10000, # Init 10000 slots to hold blocks in the HashMap
            device=device)
    return vbg

# update local tsdf from global point of view and fuse it with global tsdf
# vbg: the global tsdf representation of the scene reconstructed by far
# camera: the camera with intrinsic and extrinsic
# depth: a depth frame used for tsdf update
# depth_scale=5000.0: TUM dataset depth is scaled up by 5000.0 times
# clamping_distance: by default depth larger than 5 meters are clamped to 5 meters
def update_vbg(vbg, camera, depth, depth_scale=5000.0, clamping_distance=5.0):
    # compute the id for blocks in the frustum given depth map
    frustum_block_coords = vbg.compute_unique_block_coordinates(depth, camera.intrinsic,
                                                                camera.extrinsic, depth_scale,
                                                                clamping_distance)

    # compute TSDF and fuse it with the global one
    vbg.integrate(frustum_block_coords, depth, camera.intrinsic, camera.extrinsic,
                  depth_scale, clamping_distance)
    return vbg

# return a sorted (by timestamp) list of file paths
def get_file_list(depth_folder):
    file_list = []
    for file in os.listdir(depth_folder):
        depth_path = depth_folder + file
        file_list.append(depth_path)
    file_list.sort()

    return file_list

# A reconstruction pipeline using all the depth maps without updating
# real device pose, which makes the reconstrution results will be totally incorrect
# Just illustrating the utils for TSDF update and fusing process here
if __name__ == "__main__":
    device = o3d.core.Device('CUDA:0' if o3d.core.cuda.is_available() else 'CPU:0')
    camera = Camera()
    depth_folder = "../data/rgbd_dataset_freiburg1_xyz/depth/"
    file_list = get_file_list(depth_folder)
    vbg = create_vbg()
    debug = True

    for i in range(len(file_list)):
        print("Updating frame:", i)

        if debug and i == 4:
            break

        depth_path = file_list[i]
        depth = o3d.t.io.read_image(depth_path).to(device)

        # camera.extrinsic needs to be updated per frame, here I'm using
        # the same extrinsic just for illustration of the upate component
        vbg = update_vbg(vbg, camera, depth)

        # TODO: visualize and make sure correct
        vertex, normal = ray_cast_vbg(vbg, camera, depth)
        print(vertex.shape)
        print(normal.shape)

    # visualize the point cloud extracted from TSDF
    visualize_vbg_o3d(vbg, 180, 0, 0)
