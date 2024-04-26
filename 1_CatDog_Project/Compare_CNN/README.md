# Dog Cat Classification use CNN

This is a project to classify dog and cat images using CNN - VGG8 model.

## Classify by CNN model only

### Step 1: Load data and save data of feature
- Firstly, create a folder named "data" in this folder (at the same level as the README file) to store compressed data.
- Run file load_data to load data and save data

### Step 2: Training model
- Run file training_model.py to training CNN model for cat-dog classification
- Specifically, you need to adjust the following values when running this program:
    - Adjust parameters in the mix_data function for retrieving data. The mix_data function takes 2 parameters: an array of data and the corresponding label.
    - Adjust the model name and the number of epochs when using the train_and_save_model function.
    - Adjust the structure of the model in the build_vgg11_model function. I have already predefined two models, VGG8 and VGG11. You can customize these two models by adding or removing layers, or by defining entirely new models.

## Classify by SVM model with feature vector export from VGG8 model

### Step 1: Extract feature
- Firstly, create a folder named "extracted_1024dims_data" and "extracted_4096dims_data" in data folder.
- Adjust the model_name to choose the model you want to use to extract
- Adjust output_dir to the true folder
- Run file extract_feature.py and enter the correct sequence number corresponding to the FC layer you want to extract.

### Step 2: Training model
- Run file CNN_SVM_training.py to training CNN model for cat-dog classification
- Specifically, you need to adjust the following values when running this program:
    - Adjust parameters in the mix_data function for retrieving data. The mix_data function takes 2 parameters: an array of data and the corresponding label.
    - Adjust the model name and the number of epochs when using the train_and_save_model function.
    - Adjust data_folder to be the place to store the extracted data.