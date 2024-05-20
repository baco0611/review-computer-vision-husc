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


def train_and_save_PCA(data_dims):
    print(f"\n\n\nTraining for {data_dims}-dims vector")

    for PCA_dims in range(200, 600, 100):
        print(f"\nTraining PCA model with {PCA_dims} dims")
        model_name = "20240502_PCA_" + str(PCA_dims) + "_" + str(data_dims)

        data_folder = f"extracted_{data_dims}dims_data"
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

        print(f"Loading data ...")
        cat_data_original = read_and_append_data(cat_folders)
        dog_data_original = read_and_append_data(dog_folders)

        print(f"Training model ...")
        all_data_original = concatenate_data(cat_data_original + dog_data_original)
        model = apply_pca(all_data_original, PCA_dims)
        joblib.dump(model, f"./data/{model_name}.joblib")

        sub_folder_path = os.path.join("./data/PCA", str(PCA_dims))
        if not os.path.exists(sub_folder_path):
            os.makedirs(sub_folder_path)

        print(f"Extract cat data feature ...")
        for folder, data in zip(cat_folders, cat_data_original):
            folder_list = folder.split("/")
            name = folder_list[-1]
            cnn_dims = folder_list[2].split("_")[1]

            result = model.fit_transform(data)
            joblib.dump(result, f"./data/PCA/{PCA_dims}/{cnn_dims}_{name}")

        print(f"Extract dog data feature ...")
        for folder, data in zip(dog_folders, dog_data_original):
            folder_list = folder.split("/")
            name = folder_list[-1]
            cnn_dims = folder_list[2].split("_")[1]

            result = model.transform(data)
            joblib.dump(result, f"./data/PCA/{PCA_dims}/{cnn_dims}_{name}")

# Define folder paths
# PCA_dims = 500
data_dims = 4096
train_and_save_PCA(data_dims)
data_dims = 1024
train_and_save_PCA(data_dims)