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

    print(type(data))
    print(len(labels))

    data = np.array(data)
    labels = to_categorical(labels, num_classes=2)
    
    # Thực hiện chuẩn hóa dữ liệu ở đây nếu cần
    # Ví dụ: data = data.astype('float32') / 255.0
    
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
    # Khởi tạo mô hình và thêm các lớp giống như trong đoạn code ban đầu của bạn

    #Lenet8 lớp
    # model = Sequential()
    # model.add(Input(shape=(224, 224, 3)))
    # model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"))
    # model.add(MaxPool2D(pool_size=(2, 2)))

    # model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"))
    # model.add(MaxPool2D(pool_size=(2, 2)))

    # model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
    # model.add(MaxPool2D(pool_size=(2, 2)))

    # model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
    # model.add(MaxPool2D(pool_size=(2, 2)))

    # model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
    # model.add(MaxPool2D(pool_size=(2, 2)))

    # model.add(Flatten())
    # model.add(Dense(units=4096, activation="relu"))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=1024, activation="relu"))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=2, activation="softmax"))


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
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))

    return model

def train_and_save_model(train_x, train_y, test_x, test_y, model_name, epochs=30):
    model = build_vgg11_model()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])
    history = model.fit(train_x, train_y, batch_size=64, epochs=epochs)
    
    # Lưu mô hình
    model.save(f"./data/{model_name}_CNN_model.h5")
    
    # Quá trình huấn luyện có thể được vẽ đồ thị loss và accuracy ở đây

    # Vẽ đồ thị loss và accuracy
    plot_history(history, model_name)
    
    # Đánh giá mô hình và vẽ confusion matrix
    evaluate_and_confusion_matrix(model, test_x, test_y, model_name)


# Hàm vẽ đồ thị loss và accuracy
def plot_history(history, model_name):
    plt.figure(figsize=(12, 6))
    
    # Vẽ đồ thị loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Vẽ đồ thị accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy', color='red')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Lưu hình ảnh
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
    "./data/cat_process.joblib",
]
dog_folders = [
    "./data/dog_regular.joblib",
    "./data/dog_neg.joblib",
    "./data/dog_resize.joblib",
    "./data/dog_rotate.joblib",
    "./data/dog_process.joblib",
]

# cat_regular_x, cat_regular_y = mix_data([cat_folders[0]], 0)
# dog_regular_x, dog_regular_y = mix_data([dog_folders[0]], 1)

# train_x, test_x, train_y, test_y = process_data([(cat_regular_x, cat_regular_y), (dog_regular_x, dog_regular_y)])
# train_and_save_model(train_x, train_y, test_x, test_y, "VGG11_CatDog_Regular", epochs=30)

# cat_regular_x, cat_regular_y = mix_data([cat_folders[0], cat_folders[1]], 0)
# dog_regular_x, dog_regular_y = mix_data([dog_folders[0], dog_folders[1]], 1)

# train_x, test_x, train_y, test_y = process_data([(cat_regular_x, cat_regular_y), (dog_regular_x, dog_regular_y)])
# train_and_save_model(train_x, train_y, test_x, test_y, "VGG11_CatDog_Neg", epochs=30)

# cat_regular_x, cat_regular_y = mix_data([cat_folders[0], cat_folders[2]], 0)
# dog_regular_x, dog_regular_y = mix_data([dog_folders[0], dog_folders[2]], 1)

# train_x, test_x, train_y, test_y = process_data([(cat_regular_x, cat_regular_y), (dog_regular_x, dog_regular_y)])
# train_and_save_model(train_x, train_y, test_x, test_y, "VGG11_CatDog_Rez", epochs=30)

# cat_regular_x, cat_regular_y = mix_data([cat_folders[0], cat_folders[3]], 0)
# dog_regular_x, dog_regular_y = mix_data([dog_folders[0], dog_folders[3]], 1)

# train_x, test_x, train_y, test_y = process_data([(cat_regular_x, cat_regular_y), (dog_regular_x, dog_regular_y)])
# train_and_save_model(train_x, train_y, test_x, test_y, "VGG11_CatDog_Rot", epochs=30)

# cat_regular_x, cat_regular_y = mix_data([cat_folders[0], cat_folders[1], cat_folders[2]], 0)
# dog_regular_x, dog_regular_y = mix_data([dog_folders[0], dog_folders[1], dog_folders[2]], 1)

# train_x, test_x, train_y, test_y = process_data([(cat_regular_x, cat_regular_y), (dog_regular_x, dog_regular_y)])
# train_and_save_model(train_x, train_y, test_x, test_y, "VGG11_CatDog_NegRez", epochs=30)

# cat_regular_x, cat_regular_y = mix_data([cat_folders[0], cat_folders[1], cat_folders[3]], 0)
# dog_regular_x, dog_regular_y = mix_data([dog_folders[0], dog_folders[1], dog_folders[3]], 1)

# train_x, test_x, train_y, test_y = process_data([(cat_regular_x, cat_regular_y), (dog_regular_x, dog_regular_y)])
# train_and_save_model(train_x, train_y, test_x, test_y, "VGG11_CatDog_NegRot", epochs=30)

# cat_regular_x, cat_regular_y = mix_data([cat_folders[0], cat_folders[1], cat_folders[3], cat_folders[2]], 0)
# dog_regular_x, dog_regular_y = mix_data([dog_folders[0], dog_folders[1], dog_folders[3], dog_folders[2]], 1)

# train_x, test_x, train_y, test_y = process_data([(cat_regular_x, cat_regular_y), (dog_regular_x, dog_regular_y)])
# train_and_save_model(train_x, train_y, test_x, test_y, "VGG11_CatDog_Full", epochs=30)

cat_regular_x, cat_regular_y = mix_data([cat_folders[0]], 0)
dog_regular_x, dog_regular_y = mix_data([dog_folders[0]], 1)

train_x, test_x, train_y, test_y = process_data([(cat_regular_x, cat_regular_y), (dog_regular_x, dog_regular_y)])
# train_and_save_model(train_x, train_y, test_x, test_y, "VGG11", epochs=30)
model = load_model("./data/VGG11_CNN_model.h5")

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
            xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig("./img/VGG11.png")
plt.show()
