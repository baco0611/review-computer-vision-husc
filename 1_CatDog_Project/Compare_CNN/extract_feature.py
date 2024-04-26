# Chương trình này dùng để trích xuất đặc trưng từ các Fully Connected Layers làm đầu vào training SVM
import joblib
from keras.models import Model, load_model
from Processing_function import resize_list_image
import os

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
    images = processing_data(folders)

    features_vectors = model.predict(images)

    return features_vectors

model_name = "20240425_VGG8_Full_1"
model = define_extract_model(model_name)

cat_folders = [
    "./data/cat_regular.joblib",
    "./data/cat_neg.joblib",
    "./data/cat_resize.joblib",
    "./data/cat_rotate.joblib",
    "./data/cat_process.joblib",
    "./data/cat_flip.joblib",
]
dog_folders = [
    "./data/dog_regular.joblib",
    "./data/dog_neg.joblib",
    "./data/dog_resize.joblib",
    "./data/dog_rotate.joblib",
    "./data/dog_process.joblib",
    "./data/dog_flip.joblib",
]

output_dir = './data/extracted_1024dims_data'
# output_dir = './data/extracted_4096dims_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Duyệt qua từng file joblib của mèo, thực hiện extract và lưu kết quả
for cat_folder in cat_folders:
    cat_features = extract_feature([cat_folder])
    filename, _ = os.path.splitext(os.path.basename(cat_folder))
    output_path = os.path.join(output_dir, filename + '_features.joblib')
    joblib.dump(cat_features, output_path)
    print(f"Saved features for {cat_folder} to {output_path}")

# Duyệt qua từng file joblib của chó, thực hiện extract và lưu kết quả
for dog_folder in dog_folders:
    dog_features = extract_feature([dog_folder])
    filename, _ = os.path.splitext(os.path.basename(dog_folder))
    output_path = os.path.join(output_dir, filename + '_features.joblib')
    joblib.dump(dog_features, output_path)
    print(f"Saved features for {dog_folder} to {output_path}")