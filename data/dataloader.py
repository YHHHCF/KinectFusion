import numpy as np
import open3d as o3d
import cv2
import os
import matplotlib.pyplot as plt
from camera import *

# deg: angle in degree
# return: angle in radian
def deg2rad(deg):
    return deg * np.pi / 180

# point_cloud_o3d: o3d format point cloud
# theta_x: rotation angle around axis x
# theta_y: rotation angle around axis y
# theta_z: rotation angle around axis z
def transform_point_cloud(point_cloud_o3d, theta_x, theta_y, theta_z):
    rad_x = deg2rad(theta_x)
    rad_y = deg2rad(theta_y)
    rad_z = deg2rad(theta_z)

    # rotate theta_x degrees around x
    point_cloud_o3d.transform([[1, 0, 0, 0],
                               [0, np.cos(rad_x), -np.sin(rad_x), 0],
                               [0, np.sin(rad_x), np.cos(rad_x), 0],
                               [0, 0, 0, 1]])

    # rotate theta_y degrees around y
    point_cloud_o3d.transform([[np.cos(rad_y), 0, np.sin(rad_y), 0],
                               [0, 1, 0, 0],
                               [-np.sin(rad_y), 0, np.cos(rad_y), 0],
                               [0, 0, 0, 1]])

    # rotate theta_z degrees around z
    point_cloud_o3d.transform([[np.cos(rad_z), -np.sin(rad_z), 0, 0],
                               [np.sin(rad_z), np.cos(rad_z), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

# visualize point cloud extracted from o3d voxel block grid (TSDF representation)
def visualize_vbg_o3d(vbg, theta_x=0.0, theta_y=0.0, theta_z=0.0):
    point_cloud_o3d = vbg.extract_point_cloud()
    print(point_cloud_o3d)
    transform_point_cloud(point_cloud_o3d, theta_x, theta_y, theta_z)
    o3d.visualization.draw([point_cloud_o3d])

# point_cloud_o3d: o3d format point cloud
def visualize_point_cloud_o3d(point_cloud_o3d, theta_x=0.0, theta_y=0.0, theta_z=0.0):
    print(point_cloud_o3d)
    transform_point_cloud(point_cloud_o3d, theta_x, theta_y, theta_z)
    o3d.visualization.draw_geometries([point_cloud_o3d])

# point_cloud_numpy: a numpy array of N 3D points with shape (N, 3)
def visualize_point_cloud_numpy(point_cloud_numpy, theta_x=0.0, theta_y=0.0, theta_z=0.0):
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_numpy)
    visualize_point_cloud_o3d(point_cloud_o3d, theta_x, theta_y, theta_z)

# load depth and apply bilateral filter on it
def load_depth_as_numpy(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    depth /= 5000.0 # TUM dataset depth scale factor on depth value

    # TODO: tune filter parameters
    # currently set to diameter = 5, sigma_color = 10, sigma_space = 10
    # Filtering happens on disparity field
    depth += 1e-6
    disparity = 1 / depth
    disparity = cv2.bilateralFilter(disparity, 5, 50, 50)
    depth = 1 / disparity

    return depth

# return a sorted (by timestamp) list of file paths
def get_file_list(depth_folder):
    file_list = []
    timestamps = []

    for file in os.listdir(depth_folder):
        # drop the '.png' and convert to float
        timestamps.append(float(file[:-4]))

        # append file path
        depth_path = depth_folder + file
        file_list.append(depth_path)

    timestamps.sort()
    file_list.sort()

    return file_list, timestamps

if __name__ == "__main__":
    point_cloud_path = "rgbd_dataset_freiburg1_xyz/pointcloud_all.ply"
    rgb_path = "rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png"
    depth_path = "rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png"

    point_cloud_ply_o3d = o3d.io.read_point_cloud(point_cloud_path)
    rgb_o3d = o3d.io.read_image(rgb_path)

    depth_numpy = load_depth_as_numpy(depth_path)
    depth_o3d = o3d.geometry.Image(depth_numpy)
    rgbd_o3d = o3d.geometry.RGBDImage.create_from_tum_format(rgb_o3d, depth_o3d)

    camera = Camera()

    # 1. Use o3d to read and visualize point cloud from ply file
    visualize_point_cloud_o3d(point_cloud_ply_o3d, -90, -90, 0)

    # 2. Use o3d to create (color) point cloud from rgb and depth map
    color_point_cloud_1_frame_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, camera.o3d_intrinsic)
    visualize_point_cloud_o3d(color_point_cloud_1_frame_o3d, 180, 0, 0)

    # 3. Use o3d to create (no color) point cloud from depth map
    point_cloud_1_frame_o3d = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, camera.o3d_intrinsic)
    visualize_point_cloud_o3d(point_cloud_1_frame_o3d, 180, 0, 0)

    # 4. Generate numpy point cloud (implemented by myself) and visualize the numpy point cloud
    point_cloud_numpy = camera.point_cloud_from_depth(depth_numpy)
    visualize_point_cloud_numpy(point_cloud_numpy, 180, 0, 0)

    # 5. Compare numpy point cloud and o3d point cloud results
    # The numpy point cloud is consistant with input depth
    # The o3d point cloud is off by a sign of 1, -1, -1 on x, y, z dimensions
    print(np.mean(depth_numpy[depth_numpy>0]))
    print(np.mean(point_cloud_numpy[:,2]))
    print(np.mean(np.asarray(point_cloud_1_frame_o3d.points)[:,2]))
    print(np.mean(np.asarray(point_cloud_1_frame_o3d.points) / point_cloud_numpy, axis=0))
