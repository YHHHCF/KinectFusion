# KinectFusion
Scripts to evalute trajectory, downloaded from https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation

# 1. associate (associate 2 files according to timestamp)
python2 associate.py input1.txt input2.txt > output.txt

# 2. evaluate ate (evaluate ate of 2 trajectories)
python2 evaluate_ate.py traj1.txt traj2.txt

# 3. evaluate rpe (evaluate rpe of 2 trajectories)
python2 evaluate_rpe.py traj1.txt traj2.txt

# 4. plot_trajectory_into_image ()

