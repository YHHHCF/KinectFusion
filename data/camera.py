# Reference: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
# "Note: We recommend to use the ROS default parameter set (i.e., without undistortion),
# as undistortion of the pre-registered depth images is not trivial."
# Given the above contexts, ROS default of color camera are used for all the data

import open3d as o3d
import numpy as np

class Camera:
    def __init__(self):
        fx = 525.0
        fy = 525.0
        cx = 319.5
        cy = 239.5

        self.K = np.eye(3)
        self.K[0][0] = fx
        self.K[1][1] = fy
        self.K[0][2] = cx
        self.K[1][2] = cy

        self.z_factor = 1 / 5000.0 # TUM dataset depth scale factor on Z value

        self.o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.o3d_intrinsic.set_intrinsics(640, 480, fx, fy, cx, cy)

    def point_cloud_from_depth(self, depth_numpy):
        h, w = depth_numpy.shape
        
        # Init X and Y with u and v values
        X, Y = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
        Z = depth_numpy.astype(np.float32) * self.z_factor # Z stores z values

        # X = (u - cx) * Z / fx
        X = (X - self.K[0][2]) * Z / self.K[0][0]

        # Y = (v - cy) * Z / fy
        Y = (Y - self.K[1][2]) * Z / self.K[1][1]

        points_cloud_numpy = np.stack((X, Y, Z), axis=2) # (h, w, 3)
        points_cloud_numpy = points_cloud_numpy.reshape((h*w, 3)) # (h*w, 3)
        points_cloud_numpy = points_cloud_numpy[points_cloud_numpy[:, 2] > 0] # filter out invalid depths
        
        print("Depth map shape:", depth_numpy.shape)
        print("Point cloud shape:", points_cloud_numpy.shape)
        return points_cloud_numpy


if __name__ == "__main__":
    depth_path = "rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png"
    depth_numpy = np.asarray(o3d.io.read_image(depth_path))
    camera = Camera()
    point_cloud_numpy = camera.point_cloud_from_depth(depth_numpy)
    print(point_cloud_numpy.shape)
