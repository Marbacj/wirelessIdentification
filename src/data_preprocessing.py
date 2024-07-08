import os
import numpy as np
from CSIKit.reader import IWLBeamformReader
from CSIKit.util import csitools
from scipy.signal import butter, filtfilt
from CSIKit.csi import IWLCSIFrame

def read_csi_data(file_path):
    """读取单个.dat文件的CSI数据"""
    reader = IWLBeamformReader()
    csi_data = reader.read_file(file_path, scaled=True)
    csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude")
    csi_matrix_first = csi_matrix[:, :, 0, 0]
    return csi_matrix_first, no_frames, no_subcarriers

def remove_long_distance_multipath(csi_data, threshold=0.5):
    """移除远距离多径成分"""
    cir = np.fft.ifft(csi_data, axis=0)
    cir[int(threshold * len(cir)):] = 0
    csi_data_filtered = np.fft.fft(cir, axis=0)
    return csi_data_filtered

def remove_high_frequency_noise(csi_data, lowcut=0.3, highcut=2.0, fs=1000, order=5):
    """移除高频噪声"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    csi_data_filtered = filtfilt(b, a, csi_data, axis=0)
    return csi_data_filtered

def preprocess_csi_data(csi_data, no_frames, no_subcarriers):
    """预处理CSI数据，消除远距离多径噪声和高频噪声"""
    csi_data_filtered = remove_long_distance_multipath(csi_data)
    csi_data_filtered = remove_high_frequency_noise(csi_data_filtered)
    return csi_data_filtered

def read_and_preprocess_data(folder_path):
    """读取文件夹中的所有.dat文件并进行预处理"""
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.dat'):
            file_path = os.path.join(folder_path, file_name)
            csi_data, no_frames, no_subcarriers = read_csi_data(file_path)
            processed_data = preprocess_csi_data(csi_data, no_frames, no_subcarriers)
            all_data.append(processed_data)

    # 保存预处理后的数据到文件
    np.save('preprocessed_data.npy', all_data)

    return all_data

if __name__ == "__main__":
    folder_path = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/data/scoliosis'
    all_processed_data = read_and_preprocess_data(folder_path)
    # 将预处理后的数据保存或进一步处理
