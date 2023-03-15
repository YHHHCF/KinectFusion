# Reference: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
# "Note: We recommend to use the ROS default parameter set (i.e., without undistortion),
# as undistortion of the pre-registered depth images is not trivial."
# Given the above contexts, ROS default of color camera are used for all the data

import open3d as o3d
import numpy as np
from dataloader import *

class Camera:
    def __init__(self):
        fx = 525.0
        fy = 525.0
        cx = 319.5
        cy = 239.5

        # Used by self implemented point_cloud_from_depth() (based on numpy)
        self.K = np.eye(3)
        self.K[0][0] = fx
        self.K[1][1] = fy
        self.K[0][2] = cx
        self.K[1][2] = cy

        # Used by demo o3d point cloud visualization in dataloader.py
        self.o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.o3d_intrinsic.set_intrinsics(640, 480, fx, fy, cx, cy)


        # Used by the KinFu pipeline
        self.intrinsic = o3d.core.Tensor([[fx,  0.0, cx ],
                                          [0.0, fy,  cy ],
                                          [0.0, 0.0, 1.0]])

        # Used by the KinFu pipeline
        self.extrinsic = o3d.core.Tensor([[1.0, 0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0, 1.0]])

    def point_cloud_from_depth(self, Z):
        h, w = Z.shape

        # Init X and Y with u and v values
        X, Y = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))

        # X = (u - cx) * Z / fx
        X = (X - self.K[0][2]) * Z / self.K[0][0]

        # Y = (v - cy) * Z / fy
        Y = (Y - self.K[1][2]) * Z / self.K[1][1]

        points_cloud_numpy = np.stack((X, Y, Z), axis=2) # (h, w, 3)
        points_cloud_numpy = points_cloud_numpy.reshape((h*w, 3)) # (h*w, 3)
        points_cloud_numpy = points_cloud_numpy[points_cloud_numpy[:, 2] > 0] # filter out invalid depths

        print("Depth map shape:", Z.shape)
        print("Point cloud shape:", points_cloud_numpy.shape)
        return points_cloud_numpy


if __name__ == "__main__":
    depth_path = "rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png"
    depth_numpy = load_depth_as_numpy(depth_path)
    camera = Camera()
    point_cloud_numpy = camera.point_cloud_from_depth(depth_numpy)
