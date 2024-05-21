from sklearn import svm
import joblib
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from Processing_function import resize_list_image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# Hàm tải dữ liệu
def load_data_from_folder(folder_path):
    return joblib.load(folder_path)

def process_data(cases):
    data = []
    labels = []
    for (images, label) in cases:
        data.extend(images)
        labels.extend(label)

    print(type(data))
    # print(data[0])
    print(len(data[0]))
    print(len(labels))

    data = np.array(data)
    # labels = to_categorical(labels, num_classes=2)
    
    # return train_test_split(data, labels, test_size=0.3, random_state=42)
    return data, labels


# Hàm trộn dữ liệu từ nhiều folder
def mix_data(folders, label):
    images = []
    for folder in folders:
        data = load_data_from_folder(folder)
        if len(data) > 0:  # Kiểm tra xem thư mục có chứa ít nhất một hình ảnh không
            images.append(data)  # Chỉ thêm dữ liệu vào danh sách nếu thư mục không trống

    images = np.array(images)
    images = images.squeeze()
    return images, [label] * len(images)


labels = joblib.load("../dataset/data/label.joblib")

def validate(PCA_dims, feature_dims, name, model, index):
    print(PCA_dims, feature_dims, name)

    folders = [
        f"./data/PCA/{PCA_dims}/{feature_dims}dims_raw_image_features.joblib",
        f"./data/PCA/{PCA_dims}/{feature_dims}dims_negative_image_features.joblib",
        f"./data/PCA/{PCA_dims}/{feature_dims}dims_resized_image_features.joblib",
        f"./data/PCA/{PCA_dims}/{feature_dims}dims_rotated_image_features.joblib",
        f"./data/PCA/{PCA_dims}/{feature_dims}dims_process_image_features.joblib",
    ]

    data = joblib.load(folders[index])
    print(len(data))

    predictions = model.predict(data)
    conf_matrix = confusion_matrix(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    print(f"{name} Accuracy:", accuracy)
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(f"./img/validation/PCA/{model_name}_{name}_confuse_matrix.png")
    plt.close()

date = "20240521"
feature_dims = 1024

for x in range(200, 600, 100):
    model_name = f"{date}_SVM_{feature_dims}_{x}"
    model = joblib.load("./data/" + model_name + ".joblib")
    i = 0
    for y in ["raw", "negative", "resized", "rotated", "flipped"]:
        validate(x, feature_dims, y, model, i)
        i+=1