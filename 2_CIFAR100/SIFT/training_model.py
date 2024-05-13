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
from sklearn.model_selection import train_test_split


# Start time counting
start_time = time.time()

size = 20
codebook_name = "20240510_process_20_codebook"
data_flow = "process"
date = "20240510"
model_name = f"{date}_SVM_{data_flow}__{size}"

print("Loading data ...")
codebook = load_codebook(f"./data/{codebook_name}.joblib")
description = joblib.load(f'./data/CIFAR10/{data_flow}_description.joblib')
label = joblib.load(".././dataset/CIFAR10/train/label.joblib")
print(len(label), len(description), int(len(description)/len(label)))
# print(label[:100])
label = np.concatenate([label, label, label, label, label])
print(len(label))
# print(label[:100])

print(codebook.shape)

set_data = set()
for x in description:
    set_data.add(len(x))
print(set_data)

print("Represent data ...")
data = [represent_image_features(x, codebook, size) for x in description]


data = np.array(data)
print(data.shape)

train_set, test_set, train_y, test_y = train_test_split(data, label, test_size=0.3, random_state=42)

print("Training model ...")

model = svm.SVC(kernel='linear')  # Chọn kernel tùy ý (linear, rbf, ...)
model.fit(train_set, train_y)

print("\nSaving model ...") 
joblib.dump(model, f"./data/{model_name}.joblib")

print("\nAccuracy of Valid set and Test set")

# Dự đoán trên tập validation
# valid_preds = model.predict(valid_set)
# valid_accuracy = accuracy_score(valid_y, valid_preds)
# print("Validation Accuracy:", valid_accuracy)

# Dự đoán trên tập test
print(len(test_set))
test_preds = model.predict(test_set)
test_accuracy = accuracy_score(test_y, test_preds)
print("Test Accuracy:", test_accuracy)
# Tính toán confusion matrix
conf_matrix = confusion_matrix(test_y, test_preds)

# Vẽ confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=[], yticklabels=[], annot_kws={"size": 14})
plt.xlabel('Predicted labels', fontsize="14")
plt.ylabel('True labels', fontsize="14")
plt.title('Confusion Matrix', fontsize="14")
plt.savefig(f'./image/{model_name}_test_matrix.png')
# plt.show()

# dataset = train_set + valid_set +test_set
dataset = data
print(len(dataset))
data_preds = model.predict(dataset)
data_y = label
data_accuracy = accuracy_score(data_y, data_preds)
print("Data Accuracy:", test_accuracy)

conf_matrix = confusion_matrix(data_y, data_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=[], yticklabels=[], annot_kws={"size": 14})
plt.xlabel('Predicted labels', fontsize="14")
plt.ylabel('True labels', fontsize="14")
plt.title('Confusion Matrix', fontsize="14")
plt.savefig(f'./image/{model_name}_full_matrix.png')
# plt.show()




# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))