# pose.py implements pose estimation by adopting point to plain ICP algorithm
# reference: http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html

import open3d as o3d
import numpy as np
import sys
sys.path.append('../data/')
from camera import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    depth_path_1 = "../data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png"
    depth_path_2 = "../data/rgbd_dataset_freiburg1_xyz/depth/1305031102.194330.png"

    T_init = np.eye(4)
    camera = Camera()

    depth_numpy_1 = load_depth_as_numpy(depth_path_1)
    pc_numpy_1 = camera.point_cloud_from_depth(depth_numpy_1)
    pc_o3d_1 = o3d.geometry.PointCloud()
    pc_o3d_1.points = o3d.utility.Vector3dVector(pc_numpy_1)
    pc_o3d_1.estimate_normals()
    pc_o3d_1.normalize_normals()
    print(pc_o3d_1.has_normals())
    print(pc_o3d_1.has_points())

    depth_numpy_2 = load_depth_as_numpy(depth_path_2)
    pc_numpy_2 = camera.point_cloud_from_depth(depth_numpy_2)
    pc_o3d_2 = o3d.geometry.PointCloud()
    pc_o3d_2.points = o3d.utility.Vector3dVector(pc_numpy_2)
    pc_o3d_2.estimate_normals()
    pc_o3d_2.normalize_normals()
    print(pc_o3d_2.has_normals())
    print(pc_o3d_2.has_points())

    threshold = 0.02

    reg_p2l = o3d.pipelines.registration.registration_icp(
                pc_o3d_1, pc_o3d_1, threshold, T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())

    print(reg_p2l)
    print("Transformation between pc_o3d_1 and pc_o3d_1 is:")
    print(reg_p2l.transformation)

    reg_p2l = o3d.pipelines.registration.registration_icp(
                pc_o3d_1, pc_o3d_2, threshold, T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())

    print(reg_p2l)
    print("Transformation between pc_o3d_1 and pc_o3d_2 is:")
    print(reg_p2l.transformation)
