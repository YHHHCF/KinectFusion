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
            print(f'{self.label:s} wall time is {wall_time:.1f} s.')
        else:
            print(f'{self.label:s} Wall time is {wall_time*1000:.3f} ms.')
        self.start = 0.0


def filter_depths(data_folder="./data/rgbd_dataset_freiburg1_xyz/"):
        depth_folder = data_folder + "depth/"
        depth_name_list = os.listdir(depth_folder)
        depth_name_list.sort()

        filtered_depth_folder = data_folder + "filtered_depth/"

        if not os.path.isdir(filtered_depth_folder):
            os.mkdir(filtered_depth_folder)

        if len(os.listdir(filtered_depth_folder)) == len(os.listdir(depth_folder)):
            print("Already filtered, continue.")
            return

        for i in range(len(depth_name_list)):
            print(f'Filtering depth frame {i:d} out of {len(depth_name_list):d} frames.')
            depth_path = depth_folder + depth_name_list[i]
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
            depth /= 5000.0 # TUM dataset depth scale factor on depth value

            depth += 1e-6
            disparity = 1 / depth
            depth = cv2.bilateralFilter(depth, 15, 75, 75)
            depth = 1 / disparity
            depth -= 1e-6
            depth *= 5000

            depth = depth.astype(np.uint16)
            cv2.imwrite(filtered_depth_folder + depth_name_list[i], depth)

class KinectFusion:
    def __init__(self, data_folder="./data/rgbd_dataset_freiburg1_xyz/", depth_filtered=False):
        # current frame's index
        self.frame_id = 0

        # device to run the pipeline: CPU or CUDA
        self.device = o3d.core.Device('CUDA:0' if o3d.core.cuda.is_available() else 'CPU:0')

        # the TSDF representation
        self.vbg = create_vbg(self.device)

        self.depth_filtered = depth_filtered
        self.data_folder = data_folder

        # a list of file paths for depth maps, already sorted by timestamp
        if self.depth_filtered:
            self.depth_file_list, self.timestamps = get_file_list(self.data_folder + "filtered_depth/")
        else:
            self.depth_file_list, self.timestamps = get_file_list(self.data_folder + "depth/")
        self.num_frames = len(self.depth_file_list)

        print("KinectFusion pipeline total frame number:", self.num_frames)
        print("Depth filtered:", self.depth_filtered)

        # A point cloud contains vertices and normals

        # self.prev_point_cloud always holds point cloud estimated from
        # ray casting from the previous frame at global position
        self.prev_point_cloud = None

        # self.curr_point_cloud always holds point cloud computed from
        # surface measurement from the current frame at current (local) position
        self.curr_point_cloud = None

        # the depth camera (including: intrinsic, extrinsic, image size, utils)
        self.camera = Camera()

        # a list of poses (in trajectory) corresponding to each depth frame
        # timestamp tx ty tz qx qy qz qw
        self.poses = np.zeros((self.num_frames, 8))

    # Get the current frame's depth map
    def get_depth_map(self):
        depth_path = self.depth_file_list[self.frame_id]
        depth = o3d.t.io.read_image(depth_path)

        depth_numpy = depth.as_tensor().numpy()
        depth_numpy = depth_numpy.reshape(depth_numpy.shape[:2])

        return depth, depth_numpy

    # Save trajectory (as txt file) and TSDF
    def save_results(self, results_foler='./results/'):
        trajectory_path = results_foler + 'trajectory.txt'
        with open(trajectory_path, 'w') as f:
            f.write('# Output trajectory with each line: timestamp tx ty tz qx qy qz qw\n')
            for i in range(self.poses.shape[0]):
                for j in range(self.poses.shape[1]):
                    f.write('{:.4f} '.format(self.poses[i][j]))
                f.write('\n')
        print("Trajectory saved as:", trajectory_path)

        vbg_path = results_foler + 'TSDF.npz'
        self.vbg.save(vbg_path)
        print("TSDF saved as:", vbg_path)

    # Run the first frame, it only requires TSDF Update and Surface Prediction 
    def first_frame(self):
        print(f'========Running frame {self.frame_id:d} out of {self.num_frames:d} frames========')

        # Pose Estimation: init inital pose as identical
        self.camera.extrinsic = np.eye(4)
        self.poses[self.frame_id][0] = self.timestamps[self.frame_id]
        self.poses[self.frame_id][1:] = matrix_to_trajectory(self.camera.extrinsic)

        # TSDF Update
        depth, depth_numpy = self.get_depth_map()
        self.vbg = update_vbg(self.vbg, self.camera, depth)

        # Surface Prediction
        self.prev_point_cloud = ray_cast_vbg(self.vbg, self.camera, depth)

        self.frame_id += 1

    def has_next_frame(self):
        return self.frame_id < self.num_frames

    # Run the next (but not the first) frame
    def next_frame(self):
        print(f'========Running frame {self.frame_id:d} out of {self.num_frames:d} frames========')
        timer = PerfTimer()

        # Depth streaming
        timer.startMeasure("Depth Streaming")
        start_time = time.time()
        depth, depth_numpy = self.get_depth_map()
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
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

        self.camera.extrinsic = reg_p2l.transformation
        timer.stopMeasure()

        # Update timestamp and trajectory
        self.poses[self.frame_id][0] = self.timestamps[self.frame_id]
        self.poses[self.frame_id][1:] = matrix_to_trajectory(self.camera.extrinsic)

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
    filtered_depths = False

    if filtered_depths:
        filter_depths()

    kf = KinectFusion(depth_filtered=filtered_depths)

    kf.first_frame()

    timer = PerfTimer()

    while kf.has_next_frame():
        timer.startMeasure("Overall frame")
        kf.next_frame()
        timer.stopMeasure()

        # visualize_point_cloud_o3d(kf.curr_point_cloud, 180, 0, 0)
        if kf.frame_id == 30:
            break

    # TODO: visualize pose
    kf.save_results()

    visualize_vbg_o3d(kf.vbg, 180, 0, 0)
