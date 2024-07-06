import numpy as np

def extract_features(csi_data):
    """从预处理后的CSI数据中提取特征"""
    features = []
    for csi_sample in csi_data:
        # 计算时域和频域特征，例如最大值、最小值、标准差、能量等
        time_domain_features = compute_time_domain_features(csi_sample)
        frequency_domain_features = compute_frequency_domain_features(csi_sample)
        features.append(np.concatenate([time_domain_features, frequency_domain_features]))
    return np.array(features)

def compute_time_domain_features(csi_sample):
    # 示例：计算时域特征
    return np.array([np.mean(csi_sample), np.std(csi_sample)])

def compute_frequency_domain_features(csi_sample):
    # 示例：计算频域特征
    return np.array([np.fft.fft(csi_sample)])

if __name__ == "__main__":
    # 示例：从文件读取预处理后的CSI数据
    csi_data = np.load('preprocessed_data.npy')
    features = extract_features(csi_data)
    np.save('features.npy', features)
