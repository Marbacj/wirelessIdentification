import os
import pandas as pd
import chardet

def load_data_from_folder(folder_path):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dat')]
    df_list = []

    for file in all_files:
        with open(file, 'rb') as f:  # 以二进制模式读取文件
            result = chardet.detect(f.read())  # 检测编码
        encoding = result['encoding']  # 获取检测到的编码

        df = pd.read_csv(file, delimiter=',', encoding=encoding)  # 使用检测到的编码读取文件
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

if __name__ == "__main__":
    folder_path = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/data/scoliosis'
    combined_df = load_data_from_folder(folder_path)
    print(combined_df.head())
