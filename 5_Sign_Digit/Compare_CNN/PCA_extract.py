import joblib
import numpy as np
from sklearn.decomposition import PCA
import os

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
        folder = f"./data/{data_folder}/raw_image_features.joblib"
        folder = f"./data/{data_folder}/process_image_features.joblib"

        print(f"Loading data ...")
        data = joblib.load(folder)

        print(f"Training model ...")
        model = apply_pca(data, PCA_dims)
        joblib.dump(model, f"./data/{model_name}.joblib")

        sub_folder_path = os.path.join("./data/PCA", str(PCA_dims))
        if not os.path.exists(sub_folder_path):
            os.makedirs(sub_folder_path)

        folders = [
            f"./data/{data_folder}/raw_image_features.joblib",
            f"./data/{data_folder}/negative_image_features.joblib",
            f"./data/{data_folder}/resized_image_features.joblib",
            f"./data/{data_folder}/rotated_image_features.joblib",
            f"./data/{data_folder}/flipped_image_features.joblib",
            f"./data/{data_folder}/process_image_features.joblib",
        ]

        print(f"Extract feature ...")
        for folder in folders:
            folder_list = folder.split("/")
            name = folder_list[-1]
            cnn_dims = folder_list[2].split("_")[1]

            data = joblib.load(folder)

            result = model.fit_transform(data)
            joblib.dump(result, f"./data/PCA/{PCA_dims}/{cnn_dims}_{name}")


# Define folder paths
# PCA_dims = 500
data_dims = 4096
train_and_save_PCA(data_dims)
data_dims = 1024
train_and_save_PCA(data_dims)