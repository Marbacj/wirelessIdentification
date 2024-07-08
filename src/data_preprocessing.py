import os
import numpy as np
from CSIKit.reader import IWLBeamformReader
from CSIKit.util import csitools
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
def read_csi_data(file_path):
    """读取单个.dat文件的CSI数据"""
    reader = IWLBeamformReader()
    csi_data = reader.read_file(file_path, scaled = True)
    csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data,
                                                             metric="amplitude")
    csi_matrix_first = csi_matrix[:, :, 0, 0]
    return csi_matrix_first, no_frames, no_subcarriers

def preprocess_csi_data(csi_matrix, no_frames, no_subcarriers):
    """预处理CSI数据，消除远距离多径噪声和高频噪声"""
    # 初始化一个空列表来存储预处理后的数据
    processed_data = []

    # 遍历每一帧数据
    for frame in range(no_frames):
        # 获取当前帧的CSI矩阵
        frame_csi = csi_matrix[frame]

        # 将CFR转换为CIR（信道脉冲响应）
        cir = np.fft.ifft(frame_csi, axis=0)

        # 移除延迟超过0.5微秒的多径成分
        cir_cleaned = np.where(np.abs(cir) < 1e-5, cir, 0)  # 设置阈值以移除远距离多径
        cfr_cleaned = np.fft.fft(cir_cleaned, axis=0)

        # 应用巴特沃斯带通滤波器以消除高频噪声
        nyquist_freq = 0.5 * no_subcarriers  # 奈奎斯特频率
        lowcut = 0.3 / nyquist_freq  # 低截止频率
        highcut = 2 / nyquist_freq  # 高截止频率
        b, a = butter(1, [lowcut, highcut], btype='band')
        filtered_csi = filtfilt(b, a, cfr_cleaned, axis=0)

        # 将预处理后的数据添加到列表中
        processed_data.append(filtered_csi)

    # 将所有帧的预处理数据堆叠成一个矩阵
    processed_data = np.stack(processed_data, axis=0)

    return processed_data

# ... 其他代码保持不变 ...

def read_and_preprocess_data(folder_path):
    """读取文件夹中的所有.dat文件并进行预处理"""
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.dat'):
            file_path = os.path.join(folder_path, file_name)
            csi_data, no_frames, no_subcarriers = read_csi_data(file_path)  # 修改这里以返回所有需要的参数
            processed_data = preprocess_csi_data(csi_data, no_frames, no_subcarriers)  # 修改这里以传递所有需要的参数
            all_data.append(processed_data)

    # 保存预处理后的数据到文件
    np.save('preprocessed_data.npy', all_data)

    return all_data

# ... 其他代码保持不变 ...


if __name__ == "__main__":
    folder_path = '/Users/bachmar/wiridt/wirelessIdentification/data/scoliosis'
    all_processed_data = read_and_preprocess_data(folder_path)
    # 将预处理后的数据保存或进一步处理
