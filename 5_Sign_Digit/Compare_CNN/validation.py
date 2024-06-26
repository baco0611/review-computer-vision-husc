from keras.models import load_model
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from Processing_function import resize_list_image


# Hàm tải dữ liệu
def load_data_from_folder(folder_path):
    return joblib.load(folder_path)

def process_data(cases, size):
    data = []
    labels = []
    for (images, label) in cases:
        data.extend(resize_list_image(images))
        labels.extend(label)

    print(type(data))
    print(len(labels))

    data = np.array(data)
    labels = to_categorical(labels, num_classes=size)

    return data, labels

def mix_data(folders, label):
    images = []
    for folder in folders:
        images += load_data_from_folder(folder)
    # return images, [label] * len(images)
    return images, [label] * len(images)

folders = [
    "../dataset/data/raw_image.joblib",
    "../dataset/data/negative_image.joblib",
    "../dataset/data/resized_image.joblib",
    "../dataset/data/rotated_image.joblib",
    "../dataset/data/flipped_image.joblib",
]

# Lấy dữ liệu mong muốn
# Hàm mix_data với hai tham số ([array of dataset], label)
name = "raw"
labels = joblib.load("../dataset/data/label.joblib")
size = len(set(labels))
model_name = "20240520_VGG8_process_1"
model = load_model(f"./data/{model_name}_CNN_model.h5")


def validate(name, number):
    data = joblib.load(folders[number])
    data, label = process_data([(data, labels)], size)
    print(type(data), len(data))
    print(type(label), len(label))
    predictions = model.predict(data)

    # Chuyển đổi dự đoán và nhãn thực tế từ dạng one-hot về dạng chỉ số
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(label, axis=1)

    # Tính toán confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"{name} accuracy: {accuracy}")

    # Vẽ và hiển thị confusion matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 14})
    plt.xlabel('Predicted labels', fontsize="14")
    plt.ylabel('True labels', fontsize="14")
    plt.title('Confusion Matrix', fontsize="14")
    plt.xticks(fontsize="14")
    plt.yticks(fontsize="14")
    plt.savefig("./img/validate/" + model_name + "_" + name + ".png")
    # plt.show()


i = 0
for x in ["raw", "negative", "resized", "rotated", "flipped"]:
    validate(x, i)
    i+=1