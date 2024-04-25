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
from sklearn.metrics import confusion_matrix
import seaborn as sns
from Processing_function import resize_list_image


# Hàm tải dữ liệu
def load_data_from_folder(folder_path):
    return joblib.load(folder_path)

def process_data(cases):
    data = []
    labels = []
    for (images, label) in cases:
        data.extend(resize_list_image(images))
        labels.extend(label)

    # data = cases[0][0] + cases[1][0]
    # labels = cases[0][1] + cases[1][1]
    print(len(labels))

    data = np.array(data)
    labels = to_categorical(labels, num_classes=2)

    print(type(data))
    print(len(labels))
   
    return data, labels

def mix_data(folders, label):
    images = []
    for folder in folders:
        images += load_data_from_folder(folder)
    # return images, [label] * len(images)
    return images, [label] * len(images)

def evaluate_and_confusion_matrix(model, test_x, test_y, model_name):
    test_predictions = model.predict(test_x)
    test_predictions_labels = np.argmax(test_predictions, axis=1)
    test_true_labels = np.argmax(test_y, axis=1)
    
    conf_matrix = confusion_matrix(test_true_labels, test_predictions_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    
    # Lưu hình ảnh
    plt.savefig(f"./img/{model_name}_confusion_matrix.png")
    plt.close()

cat_folders = [
    "./data/cat_regular.joblib",
    "./data/cat_neg.joblib",
    "./data/cat_resize.joblib",
    "./data/cat_rotate.joblib",
    "./data/cat_flip.joblib",
    "./data/cat_process.joblib",
]
dog_folders = [
    "./data/dog_regular.joblib",
    "./data/dog_neg.joblib",
    "./data/dog_resize.joblib",
    "./data/dog_rotate.joblib",
    "./data/dog_flip.joblib",
    "./data/dog_process.joblib",
]

model_name = "VGG11_CatDog_Full"
model = load_model("./data/" + model_name +"_CNN_model.h5")

name = model_name + "_flip"
cat_regular_x, cat_regular_y = mix_data([cat_folders[4]], 0)
dog_regular_x, dog_regular_y = mix_data([dog_folders[4]], 1)

data, label = process_data([(cat_regular_x, cat_regular_y), (dog_regular_x, dog_regular_y)])
print(type(data), len(data))
print(type(label), len(label))

predictions = model.predict(data)

# Chuyển đổi dự đoán và nhãn thực tế từ dạng one-hot về dạng chỉ số
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(label, axis=1)

# Tính toán confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)


# Vẽ và hiển thị confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig("./img/validation/" + name + "_confuse_matrix.png")
plt.show()


