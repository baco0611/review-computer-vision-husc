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
    print(data[0])
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

# data_folder = "extracted_4096dims_data"
data_folder = "extracted_1024dims_data"
cat_folders = [
    f"./data/{data_folder}/cat_regular_features.joblib",
    f"./data/{data_folder}/cat_neg_features.joblib",
    f"./data/{data_folder}/cat_resize_features.joblib",
    f"./data/{data_folder}/cat_rotate_features.joblib",
    f"./data/{data_folder}/cat_process_features.joblib",
    f"./data/{data_folder}/cat_flip_features.joblib",
]
dog_folders = [
    f"./data/{data_folder}/dog_regular_features.joblib",
    f"./data/{data_folder}/dog_neg_features.joblib",
    f"./data/{data_folder}/dog_resize_features.joblib",
    f"./data/{data_folder}/dog_rotate_features.joblib",
    f"./data/{data_folder}/dog_process_features.joblib",
    f"./data/{data_folder}/dog_flip_features.joblib",
]

model_name = "20240426_SVM_1024_Full"
name = "process_1024"
model = joblib.load(f"./data/{model_name}.joblib")

# Hàm mix_data với hai tham số ([array of dataset], label)
dog_regular_x, dog_regular_y = mix_data([dog_folders[0]], 1)
cat_regular_x, cat_regular_y = mix_data([cat_folders[0]], 0)

data, labels = process_data([(cat_regular_x, cat_regular_y), (dog_regular_x, dog_regular_y)])
print(len(data), len(labels))

predictions = model.predict(data)
conf_matrix = confusion_matrix(labels, predictions)
    
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig(f"./img/validation/{model_name}_{name}_confuse_matrix.png")
plt.close()