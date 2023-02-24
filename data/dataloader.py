import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def deg2rad(deg):
    return deg * np.pi / 180

def transform_point_cloud(pcd, theta_x, theta_y, theta_z):
    rad_x = deg2rad(theta_x)
    rad_y = deg2rad(theta_y)
    rad_z = deg2rad(theta_z)

    # rotate theta_x degrees around x
    pcd.transform([[1, 0, 0, 0],
                   [0, np.cos(rad_x), -np.sin(rad_x), 0],
                   [0, np.sin(rad_x), np.cos(rad_x), 0],
                   [0, 0, 0, 1]])

    # rotate theta_y degrees around y
    pcd.transform([[np.cos(rad_y), 0, np.sin(rad_y), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad_y), 0, np.cos(rad_y), 0],
                   [0, 0, 0, 1]])

    # rotate theta_z degrees around z
    pcd.transform([[np.cos(rad_z), -np.sin(rad_z), 0, 0],
                   [np.sin(rad_z), np.cos(rad_z), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

def visualize_point_cloud(pcd, theta_x=0.0, theta_y=0.0, theta_z=0.0):
    print(pcd)
    print(np.asarray(pcd.points).shape)

    transform_point_cloud(pcd, theta_x, theta_y, theta_z)
    o3d.visualization.draw_geometries([pcd])

def create_rgbd_image_pcd(color_path, depth_path):
    color_raw = o3d.io.read_image(color_path)
    depth_raw = o3d.io.read_image(depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw)
    print(np.asarray(rgbd_image.color).shape)
    print(np.asarray(rgbd_image.depth).shape)
    print(np.asarray(color_raw).shape)
    print(np.asarray(depth_raw).shape)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    return rgbd_image, pcd

def create_pcd_from_depth(depth_path):
    depth_raw = o3d.io.read_image(depth_path)
    depth_array = np.asarray(depth_raw)
    print("depth map shape:", depth_array.shape)
    print("depth map number of total depths:", depth_array.shape[0] * depth_array.shape[1])
    print("depth map number of valid depths:", np.sum(depth_array > 0))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, intrinsic)
    return pcd


if __name__ == "__main__":
    point_cloud_path = "rgbd_dataset_freiburg1_xyz/pointcloud_all.ply"
    color_path = "rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png"
    depth_path = "rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png"

    pcd1 = o3d.io.read_point_cloud(point_cloud_path)
    visualize_point_cloud(pcd1, -90, -90, 0)
    rgbd_image, pcd2 = create_rgbd_image_pcd(color_path, depth_path)
    print(np.asarray(pcd2.points).shape)
    visualize_point_cloud(pcd2, 180, 0, 0)

    pcd3 = create_pcd_from_depth(depth_path)
    print(np.asarray(pcd3.points).shape)
    visualize_point_cloud(pcd3, 180, 0, 0)
