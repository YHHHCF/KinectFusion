import numpy as np
import open3d as o3d
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


if __name__ == "__main__":
    point_cloud_path = "rgbd_dataset_freiburg1_xyz/pointcloud_all.ply"
    rgb_path = "rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png"
    depth_path = "rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png"

    point_cloud_ply_o3d = o3d.io.read_point_cloud(point_cloud_path)
    rgb_o3d = o3d.io.read_image(rgb_path)
    depth_o3d = o3d.io.read_image(depth_path)
    rgbd_o3d = o3d.geometry.RGBDImage.create_from_tum_format(rgb_o3d, depth_o3d)

    depth_numpy = np.asarray(depth_o3d)

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
    # The o3d point cloud is off by a scale factor of -5
    print(np.mean(depth_numpy[depth_numpy>0]) / 5000.0) # 1.427
    print(np.mean(point_cloud_numpy[:,2])) # 1.427
    print(np.mean(np.asarray(point_cloud_1_frame_o3d.points)[:,2])) # -7.136
