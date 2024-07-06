import os
import CSIKit
import numpy as np
def read_csi_data(file_path):
    """读取单个.dat文件的CSI数据"""
    csi_data = csikit.read_file(file_path)
    return csi_data

def preprocess_csi_data(csi_data):
    """对CSI数据进行预处理，包括CIR转换和多径消除"""
    # 示例：进行CIR转换和多径消除
    cir_data = csikit.convert_to_cir(csi_data)
    processed_data = csikit.remove_multipath(cir_data, threshold=0.5)
    cfr_data = csikit.convert_to_cfr(processed_data)
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
    folder_path = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/data'
    all_processed_data = read_and_preprocess_data(folder_path)
    # 将预处理后的数据保存或进一步处理