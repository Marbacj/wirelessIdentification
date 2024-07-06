import os
import numpy as np
from scipy.signal import butter, filtfilt
from CSIKit


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def preprocess_data(csi_data, lowcut=0.3, highcut=2.0, fs=50.0):
    filtered_data = bandpass_filter(csi_data, lowcut, highcut, fs)
    return filtered_data


# 定义文件夹路径
folder_path = "C:/Users/Uncle/PycharmProjects/wirelessIdentification/data/scoliosis"

# 遍历文件夹中的每一个文件
for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        file_path = os.path.join(folder_path, filename)

        # 获取合适的读取器
        reader = get_reader(file_path)

        # 读取文件
        csi_data = reader.read_file(file_path)

        # 假设 csi_data 包含 CSI 矩阵
        csi_matrix = csi_data.csi_matrix.astype(np.int16)  # 将数据转换为 int16

        # 对 CSI 数据进行预处理
        preprocessed_data = preprocess_data(csi_matrix)

        # 处理后的数据可以保存或进一步分析
        print(f"Processed {filename}")

print("All files processed.")
