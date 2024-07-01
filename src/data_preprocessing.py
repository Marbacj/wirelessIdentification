from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """
    数据清洗和标准化
    :param df: 原始数据的 DataFrame
    :return: 清洗和标准化后的 DataFrame
    """
    df.dropna(inplace=True)  # 删除缺失值

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df.values)

    df_scaled = pd.DataFrame(scaled_features, columns=df.columns)
    return df_scaled

if __name__ == "__main__":
    from data_loader import load_data_from_folder

    folder_path = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/data/scoliosis'
    df = load_data_from_folder(folder_path)
    df_scaled = preprocess_data(df)
    print(df_scaled.head())
