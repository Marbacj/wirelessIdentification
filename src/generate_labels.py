import numpy as np

def generate_labels(num_samples, save_path):
    labels = np.random.randint(0, 2, size=num_samples)  # 随机生成0和1的标签
    np.save(save_path, labels)

if __name__ == "__main__":
    num_samples = 100  # 假设有100个样本
    save_path = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/label/npy_file.npy'  # 指定保存路径
    generate_labels(num_samples, save_path)
