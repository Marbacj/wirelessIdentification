import numpy as np
from scipy.signal import find_peaks

def detect_steps(csi_data):
    """从CSI数据中检测步长"""
    steps = []
    for csi_sample in csi_data:
        peaks, _ = find_peaks(csi_sample, height=0.5)  # 示例阈值
        steps.append(peaks)
    return steps

if __name__ == "__main__":
    csi_data = np.load('preprocessed_data.npy')
    steps = detect_steps(csi_data)
    np.save('steps.npy', steps)
