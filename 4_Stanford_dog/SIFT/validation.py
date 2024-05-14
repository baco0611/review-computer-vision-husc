from image_processing import *
from convert_keypoint_tuple import *
import time
import joblib
from bovw import *
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading file ...")

name = "Flip"

cat_dir = "../image/Cat" + name
(cat_bgr, cat_gray) = load_images(cat_dir)
(cat_keypoints, cat_description) = extract_visual_features(cat_gray)

dog_dir = "../image/Dog" + name
(dog_bgr, dog_gray) = load_images(dog_dir)
(dog_keypoints, dog_description) = extract_visual_features(dog_gray)

codebook = joblib.load("./data/codebook.joblib")
model = joblib.load("./data/SVM_model.joblib")

cat_data = [represent_image_features(x, codebook) for x in cat_description]
dog_data = [represent_image_features(x, codebook) for x in dog_description]

data = cat_data + dog_data
y = [0] * len(cat_data) + [1] * len(dog_data)

y_pred = model.predict(data)

conf_matrix = confusion_matrix(y, y_pred)

# Hiển thị ma trận nhầm lẫn bằng seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig("./image/validation/" + name + "_confuse_matrix.png")
plt.show()