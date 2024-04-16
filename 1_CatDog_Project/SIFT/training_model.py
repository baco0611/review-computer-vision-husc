from bovw import *
from image_processing import split_dataset
import joblib
import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Start time counting
start_time = time.time()

print("Loading data ...")
cat_description = joblib.load('./data/cat_description.joblib')
dog_description = joblib.load('./data/dog_description.joblib')
codebook = joblib.load("./data/codebook.joblib")

print("Splitting data ...")
# cat_train_set, cat_valid_set, cat_test_set = split_dataset(cat_description)
# dog_train_set, dog_valid_set, dog_test_set = split_dataset(dog_description)
cat_train_set, cat_test_set = split_dataset(cat_description)
dog_train_set, dog_test_set = split_dataset(dog_description)

train_set = cat_train_set + dog_train_set
train_y = [0] * len(cat_train_set) + [1] * len(dog_train_set)

# valid_set = cat_valid_set + dog_valid_set
# valid_y = [0] * len(cat_valid_set) + [1] * len(dog_valid_set)

test_set = cat_test_set + dog_test_set
test_y = [0] * len(cat_test_set) + [1] * len(dog_test_set)

# print(len(train_set), len(valid_set), len(test_set))
print(len(train_set), len(test_set))

print("Represent data ...")
train_set = [represent_image_features(x, codebook) for x in train_set]
# valid_set = [represent_image_features(x, codebook) for x in valid_set]
test_set = [represent_image_features(x, codebook) for x in test_set]

print("Training model ...")

model = svm.SVC(kernel='linear')  # Chọn kernel tùy ý (linear, rbf, ...)
model.fit(train_set, train_y)

print("\nAccuracy of Valid set and Test set")

# Dự đoán trên tập validation
# valid_preds = model.predict(valid_set)
# valid_accuracy = accuracy_score(valid_y, valid_preds)
# print("Validation Accuracy:", valid_accuracy)

# Dự đoán trên tập test
test_preds = model.predict(test_set)
test_accuracy = accuracy_score(test_y, test_preds)
print("Test Accuracy:", test_accuracy)
# Tính toán confusion matrix
conf_matrix = confusion_matrix(test_y, test_preds)

# Vẽ confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('./image/confusion_test_matrix.png')
plt.show()

# dataset = train_set + valid_set +test_set
dataset = train_set + test_set
print(len(dataset))
data_preds = model.predict(dataset)
data_y = np.concatenate((train_y, test_y))
conf_matrix = confusion_matrix(data_y, data_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('./image/confusion_matrix.png')
plt.show()


print("\nSaving model ...") 
joblib.dump(model, "./data/SVM_model.joblib")

# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))