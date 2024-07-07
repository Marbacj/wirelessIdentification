import os
import numpy as np
from CSIKit.reader import get_reader

def read_csi_data(file_path):
    """读取单个.dat文件的CSI数据"""
    reader = get_reader(file_path)
    csi_data = reader.read_file(file_path)
    return csi_data

def convert_to_cir(csi_data):
    """将CSI数据转换为CIR数据"""
    # 确保csi_data至少是二维数组
    if csi_data.ndim == 1:
        csi_data = csi_data[np.newaxis, :]
    cir_data = np.fft.ifft(csi_data, axis=-1)
    return cir_data

def remove_multipath(cir_data, threshold=0.5):
    """移除多径效应"""
    # 示例：假设cir_data是一个复数矩阵
    # 通过设置阈值来移除多径效应
    cir_data[np.abs(cir_data) < threshold] = 0
    return cir_data

def convert_to_cfr(cir_data):
    """将CIR数据转换为CFR数据"""
    # 确保cir_data至少是二维数组
    if cir_data.ndim == 1:
        cir_data = cir_data[np.newaxis, :]
    cfr_data = np.fft.fft(cir_data, axis=-1)
    return cfr_data

def preprocess_csi_data(csi_data):
    """对CSI数据进行预处理，包括CIR转换和多径消除"""
    cir_data = convert_to_cir(csi_data)
    processed_data = remove_multipath(cir_data, threshold=0.5)
    cfr_data = convert_to_cfr(processed_data)
    return cfr_data

def read_and_preprocess_data(folder_path):
    """读取文件夹中的所有.dat文件并进行预处理"""
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.dat'):
            file_path = os.path.join(folder_path, file_name)
            csi_data = read_csi_data(file_path)
            processed_data = preprocess_csi_data(csi_data)
            all_data.append(processed_data)
    return all_data

if __name__ == "__main__":
    folder_path = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/data/scoliosis'
    all_processed_data = read_and_preprocess_data(folder_path)
    # 将预处理后的数据保存或进一步处理
