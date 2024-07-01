from data_loader import load_data_from_folder
from data_preprocessing import preprocess_data
from model import train_model, evaluate_model
from sklearn.model_selection import train_test_split

def main():
    folder_path = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/data/scoliosis'
    df = load_data_from_folder(folder_path)
    df_scaled = preprocess_data(df)

    X = df_scaled.drop('label_column', axis=1)  # 修改 'label_column' 为实际标签列名
    y = df_scaled['label_column']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_size = X_train.shape[1]
    hidden_size = 50
    num_classes = len(y.unique())
    model = train_model(X_train, y_train, input_size, hidden_size, num_classes)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
