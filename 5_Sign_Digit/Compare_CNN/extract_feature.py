# Chương trình này dùng để trích xuất đặc trưng từ các Fully Connected Layers làm đầu vào training SVM
import joblib
from keras.models import Model, load_model
from Processing_function import resize_list_image
import os
import tensorflow as tf
import gc
import shutil
import numpy as np

def define_extract_model(model_name):
    model_path = "./data/" + model_name + "_CNN_model.h5"
    model = load_model(model_path)
    model.summary()

    print("\nList of layer:")
    for i in range(len(model.layers)):
        layer = model.layers[i]
        # if 'conv' not in layer.name:
        #     continue    
        print(i , layer.name , layer.output.shape)

    # Define new model to extract feature
    index = int(input("\nWhich layer do you want to extract? "))
    output = [ model.layers[index].output ]
    model = Model(inputs=model.inputs, outputs=output)

    model.summary()

    return model

def load_data_from_folder(folder_path):
    return joblib.load(folder_path)

def processing_data(folders):
    images = []
    for folder in folders:
        images += load_data_from_folder(folder)

    images = resize_list_image(images)
    return images

def extract_feature(folders):
    data = processing_data(folders)
    temp_dir = "./temp_batches"
    os.makedirs(temp_dir, exist_ok=True)
    batch_size = 5000

    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    for i in range(num_batches):
        batch_data = data[i*batch_size:(i+1)*batch_size]
        batch_file = os.path.join(temp_dir, f"batch_{i}.joblib")
        joblib.dump(batch_data, batch_file)
    
    del data
    gc.collect()

    predictions = []
    for i in range(num_batches):
        batch_file = os.path.join(temp_dir, f"batch_{i}.joblib")
        batch_data = joblib.load(batch_file)

        batch_data =  resize_list_image(batch_data)

        batch_predictions = model.predict(batch_data)
        predictions.extend(batch_predictions)
        # Giải phóng bộ nhớ sau mỗi batch
        del batch_data
        gc.collect()

    shutil.rmtree(temp_dir)
    features_vectors = np.array(predictions)
    print(len(features_vectors))
    print(len(features_vectors[0]))

    return features_vectors

model_name = "20240520_VGG8_process_1"
model = define_extract_model(model_name)

folders = [
    "../dataset/data/raw_image.joblib",
    "../dataset/data/negative_image.joblib",
    "../dataset/data/resized_image.joblib",
    # "../dataset/data/rotated_image.joblib",
    # "../dataset/data/flipped_image.joblib",
    # "../dataset/data/process_image.joblib",
]

dims = 1024
output_dir = f'./data/extracted_{dims}dims_data'
# output_dir = './data/extracted_4096dims_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# list_feature = np.empty((0, dims))
# Duyệt qua từng file joblib của mèo, thực hiện extract và lưu kết quả
for folder in folders:
    features = extract_feature([folder])
    filename, _ = os.path.splitext(os.path.basename(folder))
    output_path = os.path.join(output_dir, filename + '_features_1.joblib')
    joblib.dump(features, output_path)
    print(f"Saved features for {folder} to {output_path}")


list_feature = []
for folder in folders:
    filename, _ = os.path.splitext(os.path.basename(folder))
    output_path = os.path.join(output_dir, filename + '_features_1.joblib')
    image = joblib.load(output_path)
    list_feature += list(image)

folder = "../dataset/data/process_image.joblib"
filename, _ = os.path.splitext(os.path.basename(folder))
output_path = os.path.join(output_dir, filename + '_features_1.joblib')
joblib.dump(features, output_path)


folder1 = "../dataset/data/process_image.joblib"
filename1, _ = os.path.splitext(os.path.basename(folder1))
output_path1 = os.path.join(output_dir, filename1 + '_features_1.joblib')
data1 = joblib.load(output_path1)

folder2 = "../dataset/data/raw_image.joblib"
filename2, _ = os.path.splitext(os.path.basename(folder2))
output_path2 = os.path.join(output_dir, filename2 + '_features_1.joblib')
data2 = joblib.load(output_path2)

diff = 0
for i in range(len(data2)):
    if not np.array_equal(data1[i], data2[i]):
        diff+=1  
        print(i, end="\t")

print()
print(diff)