import pandas as pd

def load_data(file_path):
    """
    读取并解析 .dat 文件
    :param file_path: .dat 文件路径
    :return: pandas DataFrame
    """
    df = pd.read_csv(file_path, delimiter=',')  # 根据实际情况修改 delimiter
    return df

if __name__ == "__main__":
    file_path = 'path/to/your/file.dat'
    df = load_data(file_path)
    print(df.head())
