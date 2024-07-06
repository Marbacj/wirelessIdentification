import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(gait_profiles, labels):
    """训练和评估模型"""
    X_train, X_test, y_train, y_test = train_test_split(gait_profiles, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model

if __name__ == "__main__":
    gait_profiles = np.load('gait_profiles.npy')
    classpath = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/label/npy_file.npy'
    labels = np.load(classpath)  # 假设标签数据也已准备好
    model = train_model(gait_profiles, labels)
    # 保存训练好的模型
    import joblib
    joblib.dump(model, 'trained_model.pkl')
