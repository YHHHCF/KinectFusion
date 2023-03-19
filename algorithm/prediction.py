# prediction.py implements TSDF surface predition (ray casting) component
# This component takes a TSDF scene representation and a camera pose (extrinsic)
# to compute the predicted vertex and normal map from the given pose.

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Inputs: TSDF representation and camera (pose...)
# Outputs vertex and normal maps
def ray_cast_vbg(vbg, camera, depth, frustum_block_coords):
    result = vbg.ray_cast(block_coords=frustum_block_coords,
                          intrinsic=camera.intrinsic,
                          extrinsic=o3d.core.Tensor(camera.extrinsic),
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
