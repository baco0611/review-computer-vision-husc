import joblib
import time
from Processing_function import split_dataset, resize_list_image
import tensorflow as tf
from keras.utils import  to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPool2D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Start time counting
start_time = time.time()

print("Loading data ...")
cat_bgr = joblib.load('./data/cat_bgr.joblib')
dog_bgr = joblib.load('./data/dog_bgr.joblib')

print("Processing data ...")
cat_train, cat_test = split_dataset(cat_bgr)
dog_train, dog_test = split_dataset(dog_bgr)

train_x = cat_train + dog_train
test_x = cat_test + dog_test
train_y = [0] * len(cat_train) + [1] * len(dog_train)
test_y = [0] * len(cat_test) + [1] * len(dog_test)

train_x = resize_list_image(train_x)
test_x = resize_list_image(test_x)

train_y = to_categorical(train_y, num_classes=2)
test_y = to_categorical(test_y, num_classes=2)

print(len(train_x), len(train_y))
# print(train_x[0], train_y[0])

print("Define model ...")
#VGG16
# name_of_model = "VGG16"
# model = Sequential()
# model.add(Input(shape=(224, 224, 3)))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=521, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(units=4096, activation="relu"))
# model.add(Dense(units=4096, activation="relu"))
# model.add(Dense(units=2, activation="softmax"))

#LeNet
# name_of_model = "LeNet"
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
# model.add(Dense(units=1024, activation="relu"))
# model.add(Dense(units=2, activation="softmax"))

#VGG11
name_of_model = "VGG11"
model = Sequential()
model.add(Input(shape=(224, 224, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=2, activation="softmax"))

model.summary()

print("Training model ...")
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Training model by CPU")


model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])

print(len(train_x), len(train_y))

num_of_epochs = 50
H = model.fit(train_x, train_y, batch_size=64, epochs=num_of_epochs, verbose=1)

print("Saving model ...")
model.save("./data/" + name_of_model + "_CNN_model.h5")

print("Loss and Accuracy ...")
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot training loss
axes[0].plot(np.arange(0, num_of_epochs), H.history['loss'], label='training loss', color='blue')
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Plot training accuracy
axes[1].plot(np.arange(0, num_of_epochs), H.history['accuracy'], label='accuracy', color='red')
axes[1].set_title('Training Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
fig.savefig("./image/" + name_of_model + '_accuracy_loss_plot.png')

score = model.evaluate(test_x, test_y, verbose=0)
print(score)


print("Evaluation ...")
# Dự đoán nhãn cho cả tập huấn luyện và tập kiểm tra
# train_predictions = model.predict(train_x)
test_predictions = model.predict(test_x)

# Chuyển đổi dự đoán từ dạng one-hot encoding sang nhãn đơn giản
# train_predictions_labels = np.argmax(train_predictions, axis=1)
test_predictions_labels = np.argmax(test_predictions, axis=1)
# train_true_labels = np.argmax(train_y, axis=1)
test_true_labels = np.argmax(test_y, axis=1)

# Tính toán confusion matrix cho cả tập huấn luyện và tập kiểm tra
# train_conf_matrix = confusion_matrix(train_true_labels, train_predictions_labels)
test_conf_matrix = confusion_matrix(test_true_labels, test_predictions_labels)

# Cộng confusion matrix của tập huấn luyện và tập kiểm tra lại với nhau
# total_conf_matrix = train_conf_matrix + test_conf_matrix
total_conf_matrix = test_conf_matrix

# Tính toán confusion matrix
conf_matrix = total_conf_matrix

# Hiển thị confusion matrix bằng seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (Total)')
# Lưu hình ảnh
plt.savefig("./image/" + name_of_model + "_confusion_matrix.png")
# Hiển thị hình ảnh
plt.show()


# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))