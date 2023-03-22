import numpy as np
from scipy.spatial.transform import Rotation as R

# Input: pose, a 4x4 matrix representing RT4 = R4.dot(T4), first translate then rotate
# Output: a vector of [tx ty tz qx qy qz qw] to represent the pose from trajectory
def matrix_to_trajectory(RT4):
    # 3x3 R
    R3 = RT4[:3,:3]

    # 3x1 RT
    RT3 = RT4[:3,3:4]

    # 3x1 T
    T3 = np.linalg.inv(R3).dot(RT3)

    trajectory = np.zeros(7)

    # Update translation
    trajectory[:3] = T3.reshape((3))

    # Update rotation
    trajectory[3:7] = R.from_matrix([R3]).as_quat().reshape(4)

    return trajectory


# Input: a vector of [tx ty tz qx qy qz qw] to represent the pose
# Output: pose from trajectory, a 4x4 matrix representing RT4 = R4.dot(T4), first translate then rotate
def trajectory_to_matrix(trajectory):
    # 3x1 T
    T3 = trajectory[:3].reshape((3, 1))

    # 3x3 R
    R3 = R.from_quat([trajectory[3:7]]).as_matrix()

    # 3x1 RT
    RT3 = R3.dot(T3)

    # 4x4 R and T transition matrix
    RT4 = np.zeros((4, 4))
    RT4[3][3] = 1
    RT4[:3,:3] = R3
    RT4[:3,3:4] = RT3

    return RT4


def convert_line_to_floats(line):
    assert(line)
    assert(len(line) > 0)
    results = np.zeros(len(line))

    for i in range(len(line)):
        number_as_str = line[i]
        results[i] = float(number_as_str)
    return results


# timestamps: the list of gt timestamps to be searched on
# timestamp: the query timestamp
# return: whether success, the index of the timestamp in the list larger than the query timestamp
def search_timestamp(timestamps, timestamp):
    idx = np.searchsorted(timestamps, timestamp)
    if (idx < 1 or idx >= len(timestamps)):
        print("Timestamp out of range")
        return False, -1
    return True, idx


# given the timestamp and the gt trajectory, interpolate the pose at the given timestampe
# return whether success and the interpolated pose
def interpolate_pose(gt_list, timestamps, timestamp):
    success, idx = search_timestamp(timestamps, timestamp)
    if not success:
        return success, None
    interval = timestamps[idx] - timestamps[idx-1]
    alpha = (timestamp - timestamps[idx-1]) / interval

    pose1 = gt_list[timestamps[idx-1]]
    pose2 = gt_list[timestamps[idx]]
    pose = pose1 + (pose2 - pose1) * alpha

    return success, pose
