# Chuẩn hóa không gian (Spatial Normalization)
# Đồng bộ hóa chiều dài chuỗi (Temporal Resampling/Padding):

from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
LABELS = Path('E:\Report\DL\dataset\label.csv')
PROCESSED = Path('E:\Report\DL\dataset\keypoints_processed')
KEYPOINTS_PATH = Path('E:\Report\DL\dataset\keypoints')
FRAMES, JOINTS, DIMS = 60, 67, 3
_WINDOWS_INVALID_CHARS = str.maketrans({
    "/": "／",
    "\\": "＼",
    ":": "：",
    "*": "＊",
    "?": "？",
    '"': "＂",
    "<": "＜",
    ">": "＞",
    "|": "｜",
})
def read_npz(PATH):
    sequences = []
    labels = []
    files = []
    for dir in KEYPOINTS_PATH.iterdir():
        files.extend(list(dir.rglob('*.npz')))
        # print(files)

    for file in tqdm(files[:]):
        with np.load(file) as d:
            sequence = d['sequence']
            label = d['label']
            sequences.append(sequence)
            labels.append(label)
    return files, labels, sequences


# vá những frame toàn 0 liên tục ở đầu cuối
def padding_patch(sequences):
    result = []
    for sequence in sequences:
        arr = np.array(sequence)
        # all_zero_indices = np.where(np.all(arr == 0, axis=1))[0]
        non_zero_indices = np.where(~np.all(arr == 0, axis=1))[0]
        start_idx = non_zero_indices[0]
        end_idx = non_zero_indices[-1]
        if start_idx > 0:
            arr[0:start_idx] = arr[start_idx]
        if end_idx < 59:
            arr[end_idx + 1: 60] = arr[end_idx]
        # print(np.where(np.all(arr == 0, axis=1))[0])
        result.append(arr)
    return result


def spatial_normalize(sequences, left_shoulder_idx=11, right_shoulder_idx=12):
    result = []
    for sequence in sequences:
        arr = sequence.reshape(FRAMES, JOINTS, DIMS)
        left_shoulder = arr[:, left_shoulder_idx, :]
        
        right_shoulder = arr[:, right_shoulder_idx, :]
        mid_shoulder = (left_shoulder + right_shoulder) / 2.0
        # print(mid_shoulder.shape)
        mid_shoulder_expanded = np.expand_dims(mid_shoulder, axis=1)
        # print(mid_shoulder_expanded)
        arr_shifted = arr - mid_shoulder_expanded
        shoulder_widths = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
        shoulder_widths[shoulder_widths == 0] = 1.0
        shoulder_widths_expanded = np.expand_dims(shoulder_widths, axis=(1, 2))
        arr_normalized = arr_shifted / shoulder_widths_expanded
        result.append(arr_normalized)
    return result


def save_files(target_path):
    target_path = str(target_path)
    
    for file, label, sequence in zip(files, labels, sequences):
        safe_label = str(label).translate(_WINDOWS_INVALID_CHARS)
        safe_file_name = str(file.name).translate(_WINDOWS_INVALID_CHARS)
        
        label_dir = os.path.join(target_path, safe_label)
        
        os.makedirs(label_dir, exist_ok=True)
    
        full_file_path = os.path.join(label_dir, safe_file_name)
        
        np.savez_compressed(full_file_path, label=label, sequence=sequence)
        print(f"Đã lưu: {full_file_path}")

if __name__ == '__main__':
    files, labels, sequences = read_npz(KEYPOINTS_PATH)
    df = pd.DataFrame({'label': labels,'sequence': sequences})
    sequences = padding_patch(sequences)
    normalized_arrs = spatial_normalize(sequences)

    save_files(PROCESSED)
    # labels = pd.read_csv(LABELS)
    # gloss = labels['LABEL'].unique()
    # print(len(gloss))
    pass