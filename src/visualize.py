import numpy as np
import matplotlib.pyplot as plt

JOINTS, NUMS = 67, 3
full_frames = 60
sequence_path = 'E:\Report\DL\dataset/keypoints_processed/Do Thái/D0007.npz'
data = np.load(sequence_path)
sample_arr = data['sequence']
print(sample_arr.shape)
sample_arr = sample_arr.reshape((60, JOINTS, NUMS))
print(sample_arr.shape)
eps = 1e-8
def active(data):
    active_mask = (np.linalg.norm(data, axis=2) > eps)
    return active_mask
def count_active(data):
    count_active = active(data).sum(axis=1)
    return count_active
active_count = count_active(sample_arr)
full_frames = np.where(active_count == 67)[0]

FACE_CONNECTION = [
    (0,1),(1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10)
]

ARMS_CONNECTION = [
    (11,13), (13,15), (15,17), (15,19), (15,21),
    (12,14), (14,16), (16, 18), (16,20), (16, 22),
]

BODY_CONNECTION = [
    (11, 23),
    (12, 24),
    (23, 24)
]
HAND_CONNECTIONS = [
    # Cổ tay nối với gốc của 5 ngón
    (0, 1), (0, 5), (0, 17),
    (5,9), (9,13), (13,17),
    # Ngón cái (Thumb)
    (1, 2), (2, 3), (3, 4),
    # Ngón trỏ (Index Finger)
    (5, 6), (6, 7), (7, 8),
    # Ngón giữa (Middle Finger)
    (9, 10), (10, 11), (11, 12),
    # Ngón áp út (Ring Finger)
    (13, 14), (14, 15), (15, 16), # Thường là (15, 16)
    # Ngón út (Pinky)
    (17, 18), (18, 19), (19, 20)
]



left, right = 25, 46
HAND_1 = [(u + left, v + left) for (u, v) in HAND_CONNECTIONS]
HAND_2 = [(u + right, v + right) for (u, v)  in HAND_CONNECTIONS]
print(len(HAND_1))
print(len(HAND_1) + len(HAND_2) + len(ARMS_CONNECTION) + len(BODY_CONNECTION) + len(FACE_CONNECTION))
def plot_frame(data, frame_idx, title=None, figsize=(10, 7)):
    X = data[frame_idx, :, 0]
    Y = data[frame_idx, :, 1]
    colors = ["tab:blue"] * 25 + ["tab:orange"] * 21 + ["tab:green"] * 21
    plt.gca().invert_yaxis()
    for (u, v) in FACE_CONNECTION + BODY_CONNECTION + HAND_1 + HAND_2 + ARMS_CONNECTION:
        plt.plot([X[u], X[v]], [Y[u], Y[v]])
    for idx in range(JOINTS):
        # if idx not in range(25, 67):
            plt.annotate(str(idx), (X[idx], Y[idx]), fontsize=7,
                        xytext=(3,3), textcoords="offset points")
    plt.show()

# np.linspace()
lmao = np.linspace(0, 60 - 1, 10, dtype=np.int32)
lmao = [0, 10, 20, 30, 40, 50, 59]
for i in lmao:
    plot_frame(sample_arr, i)


ALL_BONE_CONNECTION = FACE_CONNECTION + ARMS_CONNECTION + BODY_CONNECTION + HAND_1 + HAND_2
print(ALL_BONE_CONNECTION)