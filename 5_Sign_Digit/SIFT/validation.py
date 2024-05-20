from image_processing import *
from convert_keypoint_tuple import *
import time
import joblib
from bovw import *
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading file ...")

size = 200
name = "process"
date = "20240516"
valid_name = "negative"
model_name = f"{date}_{name}_{size}"

codebook = joblib.load(f"./data/{date}_{name}_{size}_codebook.joblib")
labels = joblib.load("../dataset/data/label.joblib")
model = joblib.load(f"./data/SVM_{model_name}_model.joblib")

def validation(valid_name):
    data = joblib.load(f"./data/{valid_name}_description.joblib")

    data = [represent_image_features(x, codebook) for x in data]

    y_pred = model.predict(data)

    conf_matrix = confusion_matrix(labels, y_pred)
    test_accuracy = accuracy_score(labels, y_pred)
    print(f"{valid_name} Accuracy:", test_accuracy)

    # Hiển thị ma trận nhầm lẫn bằng seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig("./image/validation/" + valid_name + "_confuse_matrix.png")
    # plt.show()

for x in ["raw", "negative", "resized", "rotated", "flipped"]:
    validation(x)