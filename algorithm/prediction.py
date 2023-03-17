# prediction.py implements TSDF surface predition (ray casting) component
# This component takes a TSDF scene representation and a camera pose (extrinsic)
# to compute the predicted vertex and normal map from the given pose.

import numpy as np
import matplotlib.pyplot as plt

# Inputs: TSDF representation and camera (pose...)
# Outputs vertex and normal maps
def ray_cast_vbg(vbg, camera, depth):
    # TODO: skip frustum_block_coords by passing it in from update step, after the pipeline works
    frustum_block_coords = vbg.compute_unique_block_coordinates(depth, camera.intrinsic,
                                                                camera.extrinsic, 5000.0, 5.0)

    # TODO: tune weight_threshold, range_map_down_factor
    result = vbg.ray_cast(block_coords=frustum_block_coords,
                          intrinsic=camera.intrinsic,
                          extrinsic=camera.extrinsic,
                          width=camera.width,
                          height=camera.height,
                          render_attributes=['vertex', 'normal'],
                          depth_scale=5000.0,
                          depth_min=0.0,
                          depth_max=5.0,
                          weight_threshold=1,
                          range_map_down_factor=8)

    return result['vertex'], result['normal']
