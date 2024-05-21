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

    print(type(data))
    print(len(labels))

    data = np.array(data)
    labels = to_categorical(labels, num_classes=2)
    
    return train_test_split(data, labels, test_size=0.3, random_state=42)


# Hàm trộn dữ liệu từ nhiều folder
def mix_data(folders, label):
    images = []
    for folder in folders:
        images += load_data_from_folder(folder)
    # return images, [label] * len(images)
    return images, [label] * len(images)


# Hàm xây dựng mô hình VGG11
def build_vgg11_model(input_shape=(224, 224, 3), num_classes=2):

    #VGG8
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=2, activation="softmax"))


    #VGG11
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=2, activation="softmax"))

    model.summary()
    return model

def train_and_save_model(train_x, train_y, test_x, test_y, model_name, epochs=30):
    model = build_vgg11_model()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])
    history = model.fit(train_x, train_y, batch_size=64, epochs=epochs)
    
    model.save(f"./data/{model_name}_CNN_model.h5")
    
    plot_history(history, model_name)
    
    evaluate_and_confusion_matrix(model, test_x, test_y, model_name)


def plot_history(history, model_name):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy', color='red')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"./img/{model_name}_loss_accuracy_plot.png")
    plt.close()

# Hàm đánh giá mô hình và vẽ confusion matrix
def evaluate_and_confusion_matrix(model, test_x, test_y, model_name):
    test_predictions = model.predict(test_x)
    test_predictions_labels = np.argmax(test_predictions, axis=1)
    test_true_labels = np.argmax(test_y, axis=1)
    
    conf_matrix = confusion_matrix(test_true_labels, test_predictions_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"], annot_kws={"size": 22})
    plt.xlabel('Predicted labels', fontsize="14")
    plt.ylabel('True labels', fontsize="14")
    plt.title('Confusion Matrix', fontsize="14")
    plt.xticks(fontsize="14")
    plt.yticks(fontsize="14")
    plt.savefig(f"./img/{model_name}_confusion_matrix.png")
    plt.close()
    
cat_folders = [
    "./data/cat_regular.joblib",
    "./data/cat_neg.joblib",
    "./data/cat_resize.joblib",
    "./data/cat_rotate.joblib",
    "./data/cat_process.joblib",
]
dog_folders = [
    "./data/dog_regular.joblib",
    "./data/dog_neg.joblib",
    "./data/dog_resize.joblib",
    "./data/dog_rotate.joblib",
    "./data/dog_process.joblib",
]

# Lấy dữ liệu mong muốn
# Hàm mix_data với hai tham số ([array of dataset], label)
cat_regular_x, cat_regular_y = mix_data([cat_folders[0]], 0)
dog_regular_x, dog_regular_y = mix_data([dog_folders[0]], 1)

train_x, test_x, train_y, test_y = process_data([(cat_regular_x, cat_regular_y), (dog_regular_x, dog_regular_y)])
print(len(train_x), len(test_x))

model_name = "20240516_VGG11_raw_"
num_of_epoch = 30
train_and_save_model(train_x, train_y, test_x, test_y, model_name, epochs=num_of_epoch)
model = load_model("./data/" + model_name + "_CNN_model.h5")

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

# Nếu dữ liệu lớn có thể cần dùng `predict_generator` hoặc `predict_on_batch` để tiết kiệm bộ nhớ
predictions = model.predict(all_data)

# Chuyển đổi dự đoán và nhãn thực tế từ dạng one-hot về dạng chỉ số
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(all_labels, axis=1)

# Tính toán confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Vẽ và hiển thị confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"], annot_kws={"size": 22})
plt.xlabel('Predicted labels', fontsize="14")
plt.ylabel('True labels', fontsize="14")
plt.title('Confusion Matrix', fontsize="14")
plt.xticks(fontsize="14")
plt.yticks(fontsize="14")
plt.savefig("./img/" + model_name + ".png")
plt.show()
