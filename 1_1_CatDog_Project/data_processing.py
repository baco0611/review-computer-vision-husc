import os
import cv2
import numpy as np
from random import randint

def process_data(source_folder, destination_folder):
    # Lấy danh sách tất cả các tệp và thư mục con trong thư mục nguồn
    for root, dirs, files in os.walk(source_folder):
        for file_name in files:
            # Đường dẫn đầy đủ đến tệp nguồn
            source_file_path = os.path.join(root, file_name)
            print(source_file_path)
            
            # Đọc ảnh gốc
            image = cv2.imread(source_file_path)
            
            # Tạo thư mục đích nếu chưa tồn tại
            relative_folder = os.path.relpath(root, source_folder)
            destination_subfolder = os.path.join(destination_folder, relative_folder)
            if not os.path.exists(destination_subfolder):
                os.makedirs(destination_subfolder)
            
            # Copy ảnh sang thư mục đích
            destination_file_path = os.path.join(destination_subfolder, file_name)
            cv2.imwrite(destination_file_path, image)
            
            # Lật ảnh
            flipped_image = cv2.flip(image, 1)
            cv2.imwrite(os.path.join(destination_subfolder, "flip_" + file_name), flipped_image)
            
            # Xoay ảnh một góc ngẫu nhiên
            angle = randint(-180, 180)  # Góc xoay ngẫu nhiên trong phạm vi [-90, 90]
            rows, cols = image.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
            cv2.imwrite(os.path.join(destination_folder, f"rotate_{file_name}"), rotated_image)
            
            # Bóp méo ảnh với tỉ lệ random
            scale_factor_x = np.random.uniform(0.5, 1.5)
            scale_factor_y = np.random.uniform(0.5, 1.5)
            resized_image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y)
            cv2.imwrite(os.path.join(destination_subfolder, "resize_" + file_name), resized_image)
            
            # Chuyển ảnh thành ảnh âm
            negative_image = cv2.bitwise_not(image)
            cv2.imwrite(os.path.join(destination_subfolder, "negative_" + file_name), negative_image)

# Đường dẫn đến thư mục chứa ảnh gốc
cat_source_folder = "./Cat1000"
dog_source_folder = "./Dog1000"

# Đường dẫn đến thư mục chứa ảnh sau khi xử lý
cat_destination_folder = "./Cat_process"
dog_destination_folder = "./Dog_process"

process_data(cat_source_folder, cat_destination_folder)
process_data(dog_source_folder, dog_destination_folder)