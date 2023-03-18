import open3d as o3d
import numpy as np
import sys

sys.path.append('./data/')
from camera import *
from dataloader import *

sys.path.append('./algorithm/')
from measurement import *
from pose import *
from update import *
from prediction import *

import matplotlib.pyplot as plt
import time


class PerfTimer:
    def __init__(self):
        self.start = 0.0
        self.label = ""

    def startMeasure(self, label=""):
        self.start = time.time()
        self.label = label

    def stopMeasure(self):
        wall_time = time.time() - self.start
        if wall_time >= 1.0:
            print(f'{self.label:s} wall time is {wall_time:.3f} s.')
        else:
            print(f'{self.label:s} Wall time is {wall_time*1000:.3f} ms.')
        self.start = 0.0


class KinectFusion:
    def __init__(self, depth_folder="./data/rgbd_dataset_freiburg1_xyz/depth/"):
        # current frame's index
        self.frame_id = 0

        # device to run the pipeline: CPU or CUDA
        self.device = o3d.core.Device('CUDA:0' if o3d.core.cuda.is_available() else 'CPU:0')

        # the TSDF representation
        self.vbg = create_vbg(self.device)

        # a list of file paths for depth maps, already sorted by timestamp
        self.depth_file_list = get_file_list(depth_folder)
        self.num_frames = len(self.depth_file_list)

        print("KinectFusion pipeline total frame number:", self.num_frames)

        # A point cloud contains vertices and normals

        # self.prev_point_cloud always holds point cloud estimated from
        # ray casting from the previous frame at global position
        self.prev_point_cloud = None

        # self.curr_point_cloud always holds point cloud computed from
        # surface measurement from the current frame at current (local) position
        self.curr_point_cloud = None

        # the depth camera (including: intrinsic, extrinsic, image size, utils)
        self.camera = Camera()

        # a list of poses corresponding to each depth frame
        self.poses = np.zeros((self.num_frames, 4, 4))

    # Get the current frame's depth map
    def get_depth_map(self):
        depth_path = self.depth_file_list[self.frame_id]
        depth = o3d.t.io.read_image(depth_path).to(self.device)
        return depth

    # Run the first frame, it only requires TSDF Update and Surface Prediction 
    def first_frame(self):
        print("Running frame:", self.frame_id)

        # Pose Estimation: init inital pose as identical
        self.poses[self.frame_id] = np.eye(4)
        self.camera.extrinsic = self.poses[self.frame_id]

        # TSDF Update
        depth = self.get_depth_map()
        self.vbg = update_vbg(self.vbg, self.camera, depth)

        # Surface Prediction
        self.prev_point_cloud = ray_cast_vbg(self.vbg, self.camera, depth)

        self.frame_id += 1

    def has_next_frame(self):
        return self.frame_id < self.num_frames

    # Run the next (but not the first) frame
    def next_frame(self):
        print("Running frame:", self.frame_id)
        timer = PerfTimer()

        # Depth streaming
        timer.startMeasure("Depth streaming")
        start_time = time.time()
        depth = self.get_depth_map()
        depth_numpy = np.asarray(depth)
        depth_numpy = depth_numpy.reshape(depth_numpy.shape[:2])
        wall_time = time.time() - start_time
        timer.stopMeasure()


        # Surface Measurement
        timer.startMeasure("Surface Measurement")
        self.curr_point_cloud = point_cloud_from_depth(depth_numpy, self.camera)
        timer.stopMeasure()

        # Pose Estimation
        timer.startMeasure("Pose Estimation")
        # TODO: tune the threshold
        threshold = 0.02

        # ICP based on previous frame's pose to estimate current frame's pose
        reg_p2l = o3d.pipelines.registration.registration_icp(
                    self.prev_point_cloud, self.curr_point_cloud, threshold, self.camera.extrinsic,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane())

        self.camera.extrinsic = reg_p2l.transformation
        timer.stopMeasure()
        print("Pose of this frame is:")
        print(self.camera.extrinsic)
        print("As trajectory is:", matrix_to_trajectory(self.camera.extrinsic))

        # TODO: visualize pose

        # TSDF Update
        timer.startMeasure("TSDF Update")
        self.vbg = update_vbg(self.vbg, self.camera, depth)
        timer.stopMeasure()

        # Surface Prediction
        timer.startMeasure("Surface Prediction")
        self.prev_point_cloud = ray_cast_vbg(self.vbg, self.camera, depth)
        timer.stopMeasure()

        self.frame_id += 1

if __name__ == "__main__":
    kf = KinectFusion()
    kf.first_frame()

    timer = PerfTimer()

    while kf.has_next_frame():
        timer.startMeasure("Overall frame")
        kf.next_frame()
        timer.stopMeasure()
