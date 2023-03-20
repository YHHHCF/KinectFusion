# prediction.py implements TSDF surface predition (ray casting) component
# This component takes a TSDF scene representation and a camera pose (extrinsic)
# to compute the predicted vertex and normal map from the given pose.

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Inputs: TSDF representation and camera (pose...)
# Outputs: point cloud (includes vertex and normal maps) rendered at global pose
def ray_cast_vbg(vbg, camera, depth, from_global_pose=True):
    if from_global_pose:
        # ray casting is performed at global pose
        extrinsic = o3d.core.Tensor(np.eye(4))
    else:
        # ray casting is performed at current pose
        extrinsic = o3d.core.Tensor(camera.extrinsic)

    frustum_block_coords = vbg.compute_unique_block_coordinates(depth, camera.intrinsic,
                                                                extrinsic, 5000.0,
                                                                5.0)
    result = vbg.ray_cast(block_coords=frustum_block_coords,
                          intrinsic=camera.intrinsic,
                          extrinsic=extrinsic,
                          width=camera.width,
                          height=camera.height,
                          render_attributes=['vertex'],
                          depth_scale=5000.0,
                          depth_min=1e-3,
                          depth_max=5.0)
    point_cloud_o3d = o3d.geometry.PointCloud()

    vertex = result['vertex'].numpy()

    # for perf optimization, down sample the vertex
    vertex = vertex[::camera.down_sample_factor,::camera.down_sample_factor,:]

    vertex = vertex.reshape((-1, 3))
    point_cloud_o3d.points = o3d.utility.Vector3dVector(vertex)
    point_cloud_o3d.estimate_normals()
    point_cloud_o3d.normalize_normals()

    return point_cloud_o3d
