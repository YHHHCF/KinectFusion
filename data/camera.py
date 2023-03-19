# Reference: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
# "Note: We recommend to use the ROS default parameter set (i.e., without undistortion),
# as undistortion of the pre-registered depth images is not trivial."
# Given the above contexts, ROS default of color camera are used for all the data

import open3d as o3d
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
from dataloader import *

class Camera:
    def __init__(self):
        # Image (depth) size
        self.width = 640
        self.height = 480

        # Intrinsics
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
        self.o3d_intrinsic.set_intrinsics(self.width, self.height, fx, fy, cx, cy)


        # Used by the KinFu pipeline
        self.intrinsic = o3d.core.Tensor([[fx,  0.0, cx ],
                                          [0.0, fy,  cy ],
                                          [0.0, 0.0, 1.0]])

        # Used by the KinFu pipeline
        self.extrinsic = np.eye(4)

        # for perf optimization, down sample the vertex and normal map
        self.down_sample_factor = 10


    def point_cloud_from_depth(self, Z, filter_invalid=True, debug=False):
        h, w = Z.shape

        vertex_map_numpy = self.vertex_map_from_depth(Z)
        vertex_map_numpy = vertex_map_numpy[::self.down_sample_factor,::self.down_sample_factor,:]

        points_cloud_numpy = vertex_map_numpy.reshape((-1,3)) # (h*w, 3)

        # filter out invalid depths
        if filter_invalid:
            clamping_distance = 5
            smallest_distance = 1e-3
            points_cloud_numpy = points_cloud_numpy[points_cloud_numpy[:,2] >= smallest_distance]
            points_cloud_numpy = points_cloud_numpy[points_cloud_numpy[:,2] <= clamping_distance]

        if debug:
            print("Depth map shape:", Z.shape)
            print("Point cloud shape:", points_cloud_numpy.shape)
        return points_cloud_numpy


    def vertex_map_from_depth(self, Z):
        h, w = Z.shape
        Z = Z.astype(np.float)
        Z /= 5000.0

        # Init X and Y with u and v values
        X, Y = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))

        # X = (u - cx) * Z / fx
        X = (X - self.K[0][2]) * Z / self.K[0][0]

        # Y = (v - cy) * Z / fy
        Y = (Y - self.K[1][2]) * Z / self.K[1][1]

        vertex_map_numpy = np.stack((X, Y, Z), axis=2) # (h, w, 3)

        return vertex_map_numpy

def visualize_map(map):
    map = map.numpy()
    plt.imshow(map)
    plt.show()

# Input: pose, a 4x4 matrix representing RT4 = R4.dot(T4), first translate then rotate
# Output: a vector of [tx ty tz qx qy qz qw] to represent the pose
def matrix_to_trajectory(RT4):
    # 3x3 R
    R3 = RT4[:3,:3]

    # 3x1 RT
    RT3 = RT4[:3,3:4]

    # 3x1 T
    T3 = np.linalg.inv(R3).dot(RT3)

    trajectory = np.zeros(7)

    # Update translation
    trajectory[:3] = T3.reshape((3))

    # Update rotation
    trajectory[3:7] = R.from_matrix([R3]).as_quat().reshape(4)

    return trajectory


if __name__ == "__main__":
    depth_path = "rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png"
    depth_numpy = load_depth_as_numpy(depth_path)
    camera = Camera()
    point_cloud_numpy = camera.point_cloud_from_depth(depth_numpy)
