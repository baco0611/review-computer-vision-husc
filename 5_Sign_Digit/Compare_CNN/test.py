import joblib
import numpy as np


image_raw = joblib.load("../dataset/data/raw_image.joblib")
image_process = joblib.load("../dataset/data/process_image.joblib")

diff = 0
for i in range(len(image_raw)):
    if not np.array_equal(image_raw[i], image_process[i]):
        diff+=1   
                    
print("Raw data error:",diff, len(image_raw), len(image_process))


data_dims = 4096
data_folder = f"extracted_{data_dims}dims_data"
image_raw = joblib.load(f"./data/{data_folder}/raw_image_features.joblib")
image_process = joblib.load(f"./data/{data_folder}/process_image_features.joblib")
diff = 0
for i in range(len(image_raw)):
    if not np.array_equal(image_raw[i], image_process[i]):
        diff+=1   
                    
print("4096 dims data error:",diff, len(image_raw), len(image_process))


data_dims = 1024
data_folder = f"extracted_{data_dims}dims_data"
image_raw = joblib.load(f"./data/{data_folder}/raw_image_features_1.joblib")
image_process = joblib.load(f"./data/{data_folder}/process_image_features_1.joblib")
diff = 0
for i in range(len(image_raw)):
    if not np.array_equal(image_raw[i], image_process[i]):
        diff+=1   
                    
print("1024 dims data error:",diff, len(image_raw), len(image_process))



PCA_dims = 200
feature_dims = 4096
data_folder = f"extracted_{data_dims}dims_data"
image_raw = joblib.load(f"./data/PCA/{PCA_dims}/{feature_dims}dims_raw_image_features.joblib",)
image_process = joblib.load(f"./data/PCA/{PCA_dims}/{feature_dims}dims_process_image_features.joblib",)
diff = 0
for i in range(len(image_raw)):
    if not np.array_equal(image_raw[i], image_process[i]):
        diff+=1   
                    
print("4096 200 dims data error:",diff, len(image_raw), len(image_process))