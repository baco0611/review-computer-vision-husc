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

# Hàm đánh giá mô hình và vẽ confusion matrix
def evaluate_and_confusion_matrix(model, test_x, test_y, model_name):
    test_predictions = model.predict(test_x)

    
    conf_matrix = confusion_matrix(test_y, test_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(f"./img/{model_name}_confusion_matrix.png")
    plt.close()

cat_folders = [
    "./data/extract_data/cat_regular_features.joblib",
    "./data/extract_data/cat_neg_features.joblib",
    "./data/extract_data/cat_resize_features.joblib",
    "./data/extract_data/cat_rotate_features.joblib",
    "./data/extract_data/cat_process_features.joblib",
]
dog_folders = [
    "./data/extract_data/dog_regular_features.joblib",
    "./data/extract_data/dog_neg_features.joblib",
    "./data/extract_data/dog_resize_features.joblib",
    "./data/extract_data/dog_rotate_features.joblib",
    "./data/extract_data/dog_process_features.joblib",
]

#Lấy dữ liệu mong muốn
# Hàm mix_data với hai tham số ([array of dataset], label)
dog_regular_x, dog_regular_y = mix_data([dog_folders[0]], 1)
cat_regular_x, cat_regular_y = mix_data([cat_folders[0]], 0)

train_x, test_x, train_y, test_y = process_data([(cat_regular_x, cat_regular_y), (dog_regular_x, dog_regular_y)])
print(len(train_x), len(test_x))

model_name = "20240425_SVM_1995"
train_and_save_model(train_x, train_y, test_x, test_y, model_name)