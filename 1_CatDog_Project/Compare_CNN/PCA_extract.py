import joblib
import numpy as np
from sklearn.decomposition import PCA
import os

def read_and_append_data(folder_paths):
    data = []
    for path in folder_paths:
        data.append(joblib.load(path))
    return data

def concatenate_data(data):
    return np.concatenate(data)

def apply_pca(data, PCA_dims):
    pca = PCA(n_components=PCA_dims)
    pca.fit(data)
    return pca

# Define folder paths
# data_folder = "extracted_4096dims_data"
data_folder = "extracted_1024dims_data"
cat_folders = [
    f"./data/{data_folder}/cat_regular_features.joblib",
    f"./data/{data_folder}/cat_neg_features.joblib",
    f"./data/{data_folder}/cat_resize_features.joblib",
    f"./data/{data_folder}/cat_rotate_features.joblib",
    f"./data/{data_folder}/cat_flip_features.joblib",
    f"./data/{data_folder}/cat_process_features.joblib",
]
dog_folders = [
    f"./data/{data_folder}/dog_regular_features.joblib",
    f"./data/{data_folder}/dog_neg_features.joblib",
    f"./data/{data_folder}/dog_resize_features.joblib",
    f"./data/{data_folder}/dog_rotate_features.joblib",
    f"./data/{data_folder}/dog_flip_features.joblib",
    f"./data/{data_folder}/dog_process_features.joblib",
]

cat_data_original = read_and_append_data(cat_folders)
dog_data_original = read_and_append_data(dog_folders)

all_data_original = concatenate_data(cat_data_original + dog_data_original)

PCA_dims = 100
model_name = "20240501_PCA_" + str(PCA_dims)
model = apply_pca(all_data_original, PCA_dims)
joblib.dump(model, f"./data/{model_name}.joblib")

sub_folder_path = os.path.join("./data/PCA", str(PCA_dims))
if not os.path.exists(sub_folder_path):
    os.makedirs(sub_folder_path)

for folder, data in zip(cat_folders, cat_data_original):
    folder_list = folder.split("/")
    name = folder_list[-1]
    cnn_dims = folder_list[2].split("_")[1]

    result = model.fit_transform(data)
    joblib.dump(result, f"./data/PCA/{PCA_dims}/{cnn_dims}_{name}")

for folder, data in zip(dog_folders, dog_data_original):
    folder_list = folder.split("/")
    name = folder_list[-1]
    cnn_dims = folder_list[2].split("_")[1]

    result = model.fit_transform(data)
    joblib.dump(result, f"./data/PCA/{PCA_dims}/{cnn_dims}_{name}")