import os
import numpy as np
import joblib
from data_preprocessing import read_and_preprocess_data
from feature_extraction import extract_features
from step_detection import detect_steps
from gait_analysis import analyze_gait
from model_training import train_model


def main(data_folder, labels_file):
    """主程序"""
    # 1. 数据预处理
    preprocessed_data = read_and_preprocess_data(data_folder)
    np.save('preprocessed_data.npy', preprocessed_data)

    # 2. 特征提取
    features = extract_features(preprocessed_data)
    np.save('features.npy', features)

    # 3. 步长检测
    steps = detect_steps(preprocessed_data)
    np.save('steps.npy', steps)

    # 4. 步态分析
    gait_profiles = analyze_gait(steps, features)
    np.save('gait_profiles.npy', gait_profiles)

    # 5. 模型训练和评估
    labels = np.load(labels_file)
    model = train_model(gait_profiles, labels)

    # 保存训练好的模型
    joblib.dump(model, 'trained_model.pkl')


if __name__ == "__main__":
    data_folder = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/data'  # 替换为你的数据文件夹路径
    labels_file = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/label/npy_file.npy'  # 替换为你的标签文件路径
    main(data_folder, labels_file)
