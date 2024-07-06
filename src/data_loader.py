import os
import pandas as pd
from csikit import CSIData
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def load_csi_data_from_folder(folder_path):
    """
    读取文件夹中的所有 .dat 文件并合并为一个 DataFrame
    :param folder_path: 文件夹路径
    :return: 合并后的 pandas DataFrame
    """
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dat')]
    csi_frames = []

    for file in all_files:
        try:
            csi_data = CSIData(file)
            df = csi_data.to_dataframe()  # 使用 csikit 将 CSI 数据转换为 DataFrame
            csi_frames.append(df)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    if not csi_frames:
        return pd.DataFrame()  # 返回空的 DataFrame

    combined_csi_df = pd.concat(csi_frames, ignore_index=True)
    return combined_csi_df


def preprocess_data(df):
    """
    对数据进行预处理，包括缺失值处理、标准化和特征选择
    :param df: 原始数据 DataFrame
    :return: 预处理后的数据
    """
    # 假设最后一列是目标变量，其他列是特征
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 处理缺失值
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 特征选择
    selector = SelectKBest(score_func=f_classif, k=10)  # 选择前10个最佳特征
    X_selected = selector.fit_transform(X_scaled, y)

    return X_selected, y


if __name__ == "__main__":
    folder_path = input("Please enter the folder path: ")
    if os.path.isdir(folder_path):
        combined_csi_df = load_csi_data_from_folder(folder_path)
        if not combined_csi_df.empty:
            X, y = preprocess_data(combined_csi_df)
            print("Preprocessed data:")
            print(X[:5])
            print("Labels:")
            print(y[:5])
        else:
            print("No valid CSI data files found.")
    else:
        print("Invalid folder path.")
