import numpy as np
import random
import os

from tqdm import tqdm
from preprocessing import PROCESSED, padding_patch, FRAMES, JOINTS, DIMS
# def horizontal_flip(sequence):
#     """Lật ngang video và hoán đổi tay trái/phải."""
#     seq = sequence.copy()
    
#     # 1. Lật trục X (Vì gốc tọa độ 0,0 ở ngực nên chỉ cần lấy x = -x)
#     seq[:, :, 0] = -seq[:, :, 0]
    
#     # 2. Hoán đổi 21 điểm của tay trái (25:46) và tay phải (46:67)
#     left_hand = seq[:, 25:46, :].copy()
#     seq[:, 25:46, :] = seq[:, 46:67, :]
#     seq[:, 46:67, :] = left_hand
    
#     # 3. Hoán đổi các điểm đối xứng trên Pose (Mắt, vai, khuỷu tay, v.v.)
#     # Theo chuẩn của MediaPipe PoseLandmarker (25 điểm đầu)
#     pose_swaps = [
#         (1, 4), (2, 5), (3, 6),     # Mắt
#         (7, 8),                     # Tai
#         (9, 10),                    # Miệng
#         (11, 12), (13, 14), (15, 16), # Vai, khuỷu tay, cổ tay
#         (17, 18), (19, 20), (21, 22), # Bàn tay (trong pose)
#         (23, 24)                    # Hông
#     ]
#     for left_idx, right_idx in pose_swaps:
#         seq[:, [left_idx, right_idx], :] = seq[:, [right_idx, left_idx], :]
        
#     return seq


def rotate(sequence, angle_range=(-15, 15)):
    """Xoay cơ thể quanh gốc tọa độ (ngực)."""
    seq = sequence.copy()
    # print(seq.shape)
    angle = random.uniform(angle_range[0], angle_range[1])
    theta = np.radians(angle)
    
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], 
                                [s,  c]])
    
    # Nhân ma trận trục XY cho toàn bộ (60, 67)
    seq[:, :, :2] = np.dot(seq[:, :, :2], rotation_matrix.T)
    return seq


def translation(sequence, shift_range=(-0.1, 0.1)):
    """Dịch chuyển toàn bộ cơ thể theo trục X, Y."""
    seq = sequence.copy()
    shift_x = random.uniform(shift_range[0], shift_range[1])
    shift_y = random.uniform(shift_range[0], shift_range[1])
    
    # Chỉ cộng vào những điểm không bị mất dấu (tọa độ != 0)
    valid = seq[:, :, 0] != 0
    seq[valid, 0] += shift_x
    seq[valid, 1] += shift_y
    
    return seq


def scale(sequence, scale_range=(0.8, 1.2)):
    """Phóng to / thu nhỏ cơ thể."""
    seq = sequence.copy()
    scale_factor = random.uniform(scale_range[0], scale_range[1])
    
    # Chỉ nhân scale cho X và Y (giữ nguyên độ sâu Z)
    seq[:, :, :2] *= scale_factor
    return seq


def gaussian_noise(sequence, mean=0.0, std=0.01):
    """Thêm độ rung (noise) ngẫu nhiên vào các khớp xương."""
    seq = sequence.copy()
    noise = np.random.normal(mean, std, seq.shape)
    
    # Không thêm nhiễu vào các điểm đang bằng 0 (bị che khuất)
    valid = seq != 0
    seq[valid] += noise[valid]
    
    return seq


def spatial_mask(sequence, mask_prob=0.15):
    """Gán tọa độ của một bàn tay về 0 để ép AI học dựa trên tay còn lại."""
    seq = sequence.copy()
    
    # 15% xác suất che tay trái
    if random.random() < mask_prob:
        seq[:, 25:46, :] = 0.0
        
    # 15% xác suất che tay phải
    if random.random() < mask_prob:
        seq[:, 46:67, :] = 0.0
        
    return seq


def temporal_speed_tweak(seq, speed_range=(0.8, 1.2)):
    """
    Ngẫu nhiên thay đổi tốc độ của chuỗi frame bằng cách lấy mẫu lại (Resampling)
    """
    num_frames, num_points, num_dims = seq.shape
    speed = random.uniform(*speed_range)
    
    # Tính số lượng frame mới sau khi thay đổi tốc độ
    new_num_frames = int(num_frames / speed)
    
    # Tạo các mốc chỉ số index mới
    indices = np.linspace(0, num_frames - 1, new_num_frames).astype(int)
    new_seq = seq[indices]
    
    
    new_seq = padding_patch(new_seq, target_frames=num_frames)
    
    return new_seq

# func_list = [rotate, translation, scale, gaussian_noise, spatial_mask]
def augment_pipeline():
    """
    Hàm này được gọi khi Load từng batch dữ liệu vào Model.
    """
    # 1. Biến đổi Thời gian (Luôn làm để đảm bảo output đúng 60 frames)
    for dir in tqdm(PROCESSED.iterdir(), total=len(list(PROCESSED.iterdir())), desc="Augmenting sequences"):
        files = list(dir.rglob('*.npz'))[:10]
        for file in (files):
            with np.load(file) as d:
                sequence = d['sequence'].reshape((FRAMES, JOINTS, DIMS))
                label = d['label']
            file_name, exc = os.path.splitext(file)
            augment_func = {
                'rotate': rotate(sequence),
                'translation': translation(sequence),
                'scale': scale(sequence),
                'gaussian_noise': gaussian_noise(sequence),
                'spatial_mask': spatial_mask(sequence),
            }
            if any(file_name.endswith(i) for i in ['rotate', 'translation', 'scale', 'gaussian_noise', 'spatial_mask']):
                continue
            for func_name, func in augment_func.items():
                new_sequence = func
                new_file_name = f"{file_name}_{func_name}.npz"
                np.savez(new_file_name,  label=label, sequence=new_sequence)
                # print()
            

    # return seq

augment_pipeline()