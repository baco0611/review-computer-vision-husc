import joblib
import cv2
import os
from random import randint
import numpy as np

(x_train, y_train), (x_test, y_test) = joblib.load("../dataset/CIFAR10/raw_data.joblib")

def check_folder(destination_folder):
    subfolders = ['raw', 'flip', 'negative', 'resize', 'rotate']
    for subfolder in subfolders:
        subfolder_path = os.path.join(destination_folder, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            print(f"Created folder: {subfolder_path}")

def process_data(destination_folder, data, label):
    
    print(f"\n\nWorking on folder {destination_folder}")
    check_folder(destination_folder)

    print("Saving label data ...")
    joblib.dump(label, f"{destination_folder}/label.joblib")
    
    print("Processing image data ...")

    i = 0

    raw_list = []
    flip_list = []
    negative_list = []
    resize_list = []
    rotate_list = []

    for x in data:
        image = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{destination_folder}/raw/{i}.jpg", image)
        raw_list.append(image)

        flipped_image = cv2.flip(image, 1)
        cv2.imwrite(f"{destination_folder}/flip/{i}.jpg", flipped_image)
        flip_list.append(flipped_image)

        angle = randint(-180, 180)  # Góc xoay ngẫu nhiên trong phạm vi [-90, 90]
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        cv2.imwrite(f"{destination_folder}/rotate/{i}.jpg", rotated_image)
        rotate_list.append(rotated_image)

        scale_factor_x = np.random.uniform(0.5, 1.5)
        scale_factor_y = np.random.uniform(0.5, 1.5)
        resized_image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y)
        cv2.imwrite(f"{destination_folder}/resize/{i}.jpg", resized_image)
        resize_list.append(resized_image)

        negative_image = cv2.bitwise_not(image)
        cv2.imwrite(f"{destination_folder}/negative/{i}.jpg", negative_image)
        negative_list.append(negative_image)

        i+=1
    
    joblib.dump(raw_list, f"{destination_folder}/raw_image.joblib")
    joblib.dump(flip_list, f"{destination_folder}/flipped_image.joblib")
    joblib.dump(negative_list, f"{destination_folder}/negative_image.joblib")
    joblib.dump(resize_list, f"{destination_folder}/resized_image.joblib")
    joblib.dump(rotate_list, f"{destination_folder}/rotated_image.joblib")
    


training_folder = "../dataset/CIFAR10/train"
testing_folder = "../dataset/CIFAR10/test"

process_data(training_folder, x_train, y_train)
process_data(testing_folder, x_test, y_test)