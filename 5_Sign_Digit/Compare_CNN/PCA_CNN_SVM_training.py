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
    print(len(labels))

    data = np.array(data)
    # labels = to_categorical(labels, num_classes=2)
    
    return train_test_split(data, labels, test_size=0.3, random_state=42)


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

def train_and_save_model(train_x, train_y, test_x, test_y, model_name):
    print("Training model ...")
    
    # train_x = train_x.reshape(train_x.shape[0], -1)
    # test_x = test_x.reshape(test_x.shape[0], -1)

    model = svm.SVC(kernel='linear')  # Chọn kernel tùy ý (linear, rbf, ...)
    model.fit(train_x, train_y)

    joblib.dump(model, "./data/" + model_name + ".joblib")

    evaluate_and_confusion_matrix(model, test_x, test_y, model_name)

    return model

# Hàm đánh giá mô hình và vẽ confusion matrix
def evaluate_and_confusion_matrix(model, test_x, test_y, model_name):
    test_predictions = model.predict(test_x)

    
    conf_matrix = confusion_matrix(test_y, test_predictions)
    accuracy = accuracy_score(test_y, test_predictions)
    print("Model Accuracy:", accuracy)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 20})
    plt.xlabel('Predicted labels', fontsize="14")
    plt.ylabel('True labels', fontsize="14")
    plt.title('Confusion Matrix', fontsize="14")
    plt.xticks(fontsize="14")
    plt.yticks(fontsize="14")
    plt.savefig(f"./img/PCA/{model_name}_confusion_matrix.png")
    plt.close()


# data_folder = "extracted_1024dims_data"
PCA_dims = 500
feature_dims = 4096
date = "20240521"
model_name = f"{date}_SVM_{feature_dims}_{PCA_dims}"

print(PCA_dims, feature_dims)

folders = [
    f"./data/PCA/{PCA_dims}/{feature_dims}dims_raw_image_features.joblib",
    f"./data/PCA/{PCA_dims}/{feature_dims}dims_negative_image_features.joblib",
    f"./data/PCA/{PCA_dims}/{feature_dims}dims_resized_image_features.joblib",
    f"./data/PCA/{PCA_dims}/{feature_dims}dims_rotated_image_features.joblib",
    f"./data/PCA/{PCA_dims}/{feature_dims}dims_process_image_features.joblib",
]

#Lấy dữ liệu mong muốn
# Hàm mix_data với hai tham số ([array of dataset], label)
data = joblib.load(folders[4])
labels = joblib.load("../dataset/data/label.joblib")
process_labels = labels * 5
labels = process_labels
labels = [x for x in labels]
size = len(set(labels))
print(len(data))

train_x, test_x, train_y, test_y = process_data([(data, labels)])
print(len(train_x), len(test_x))
print(type(train_x), type(test_x))

model = train_and_save_model(train_x, train_y, test_x, test_y, model_name)

all_data = []
all_data.extend(train_x)
all_data.extend(test_x)
all_data = np.array(all_data)

all_labels = []
all_labels.extend(train_y)
all_labels.extend(test_y)
all_labels = np.array(all_labels)

print(type(all_data), len(all_data))
print(type(all_labels), len(all_labels))

predictions = model.predict(all_data)
conf_matrix = confusion_matrix(all_labels, predictions)
accuracy = accuracy_score(all_labels, predictions)
print("Model Accuracy:", accuracy)
    
plt.figure(figsize=(15, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 20})
plt.xlabel('Predicted labels', fontsize="14")
plt.ylabel('True labels', fontsize="14")
plt.title('Confusion Matrix', fontsize="14")
plt.xticks(fontsize="14")
plt.yticks(fontsize="14")
plt.savefig(f"./img/PCA/{model_name}.png")
plt.close()