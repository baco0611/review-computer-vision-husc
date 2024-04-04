import joblib
import time
from Processing_function import split_dataset, resize_list_image
from keras.utils import  to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPool2D

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
# print(train_x[0].shape, train_y[0].shape)

print("Define model ...")
#VGG16
model = Sequential()
model.add(Input(shape=(224, 224, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="sigmoid", padding="same"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="sigmoid", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="sigmoid", padding="same"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="sigmoid", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="sigmoid", padding="same"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="sigmoid", padding="same"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="sigmoid", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=521, kernel_size=(3, 3), activation="sigmoid", padding="same"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="sigmoid", padding="same"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="sigmoid", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="sigmoid", padding="same"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="sigmoid", padding="same"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="sigmoid", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=4096, activation="sigmoid"))
model.add(Dense(units=4096, activation="sigmoid"))
model.add(Dense(units=2, activation="softmax"))

model.summary()

print("Training model ...")
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])

H = model.fit(test_x, test_y, batch_size=64, epochs=20, verbose=1)

# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))