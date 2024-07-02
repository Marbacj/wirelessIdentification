import os
import pandas as pd
from CSIKit import CSIData
import sys

def load_csi_data_from_folder(folder_path):
    """
    读取文件夹中的所有 .dat 文件并合并为一个 DataFrame
    :param folder_path: 文件夹路径
    :return: 合并后的 pandas DataFrame
    """
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dat')]
    csi_frames = []

    for file in all_files:
        csi_data = CSIData(file)
        df = csi_data.to_dataframe()  # 使用 csikit 将 CSI 数据转换为 DataFrame
        csi_frames.append(df)

    combined_csi_df = pd.concat(csi_frames, ignore_index=True)
    return combined_csi_df

if __name__ == "__main__":
    folder_path = 'path/to/your/folder'
    combined_csi_df = load_csi_data_from_folder(folder_path)
    print(combined_csi_df.head())
