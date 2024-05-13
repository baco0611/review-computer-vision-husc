# Dog Cat Classification use SIFT Features and SVM Model 

This is a project to classify dog and cat images using SIFT feature extraction. Use Bag of Visual Word (BoVW) to represent images. Then classify with SVM model (Support Vector Model).

### Step 1: Load data and save data of feature
- First, create a folder named "data" in this large project folder (at the same level as the README file) to store the compressed data.
- Run file load_data to load data and save data

### Step 2: Training codebook
- Run file training_codebook.py to train the codebook

### Step 3: Training model
- Run file traning_model.py to training SVM model for cat-dog classification