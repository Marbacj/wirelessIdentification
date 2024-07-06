import os
import numpy as np


def generate_labels(data_folder):
    labels = []
    for filename in os.listdir(data_folder):
        if filename.endswith('.dat'):
            # 提取第一个连字符（-）之前的部分作为标签
            label = filename.split('-')[0]
            labels.append(label)
    return np.array(labels)


if __name__ == "__main__":
    data_folder = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/data/scoliosis'  # 替换为你的数据文件夹路径
    labels = generate_labels(data_folder)

    # 保存标签为 npy_file.npy 文件
    save_path = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/label/npy_file.npy'
    np.save(save_path, labels)
