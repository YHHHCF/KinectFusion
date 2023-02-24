Scripts to evalute trajectory, downloaded from https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation and modified to be able to run with Python3

# 1. associate (associate 2 files according to timestamp)
python3 associate.py ../data/rgbd_dataset_freiburg1_xyz/depth.txt ../data/rgbd_dataset_freiburg1_xyz/groundtruth.txt

# 2. evaluate ate (evaluate ate of 2 trajectories)
python3 evaluate_ate.py ../data/rgbd_dataset_freiburg1_xyz/groundtruth.txt ../data/rgbd_dataset_freiburg1_xyz/groundtruth.txt

# 3. evaluate rpe (evaluate rpe of 2 trajectories)
python3 evaluate_rpe.py ../data/rgbd_dataset_freiburg1_xyz/groundtruth.txt ../data/rgbd_dataset_freiburg1_xyz/groundtruth.txt

# 4. plot_trajectory_into_image (output trajectory to image for debugging)
run from data/rgbd_dataset_freiburg1_xyz/
python3 ../../eval/plot_trajectory_into_image.py rgb.txt groundtruth.txt ./trajectory/trajectory.png

# 5. generate_pointcloud (generate colored pointclouds from 1 pair of rgb, depth)
python3 generate_pointcloud.py ../data/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png ../data/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png pointcloud_1.ply

# 6. generate_registered_pointcloud (generate colored pointclouds from all pairs of rgb, depth along with trajectory)
run from data/rgbd_dataset_freiburg1_xyz/
python3 ../../eval/generate_registered_pointcloud.py rgb_less.txt depth.txt groundtruth.txt pointcloud_all.ply