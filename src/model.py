import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_model(X_train, y_train, input_size, hidden_size, num_classes, num_epochs=100, learning_rate=0.001):
    model = SimpleNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # 转换为 Tensor
        inputs = torch.tensor(X_train.values, dtype=torch.float32)
        labels = torch.tensor(y_train.values, dtype=torch.long)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test.values, dtype=torch.float32)
        labels = torch.tensor(y_test.values, dtype=torch.long)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        print(f'Accuracy: {100 * correct / total:.2f}%')


if __name__ == "__main__":
    from data_loader import load_data
    from data_preprocessing import preprocess_data

    file_path = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/data/scoliosis'
    df = load_data(file_path)
    df_scaled = preprocess_data(df)

    X = df_scaled.drop('label_column', axis=1)  # 修改 'label_column' 为实际标签列名
    y = df_scaled['label_column']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_size = X_train.shape[1]
    hidden_size = 50
    num_classes = len(y.unique())
    model = train_model(X_train, y_train, input_size, hidden_size, num_classes)
    evaluate_model(model, X_test, y_test)
