import cv2
import os
import glob
import random

def load_images(image_dir):
    image_paths = sorted(glob.glob(os.path.join(image_dir,'*.jpg')))
    bgr_images = []
    gray_images = []
    for image_path in image_paths:
        bgr = cv2.imread(image_path)
        # cv2.imshow("Image", bgr)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bgr_images.append(bgr)
        gray_images.append(gray)
    return bgr_images, gray_images

def split_dataset(dataset, train_ratio=0.7):
    """
    Chia dataset thành ba phần train/validation/test theo tỷ lệ đã cho.
    
    Args:
    - dataset: List chứa dữ liệu cần chia.
    - train_ratio: Tỷ lệ phần train, mặc định là 0.7.
    - valid_ratio: Tỷ lệ phần validation, mặc định là 0.2.
    - test_ratio: Tỷ lệ phần test, mặc định là 0.1.
    
    Returns:
    - train_set: List chứa phần train.
    - valid_set: List chứa phần validation.
    - test_set: List chứa phần test.
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    
    # Đảm bảo tỷ lệ không bị làm tròn lên
    assert train_size + test_size == total_size
    
    # Xáo trộn dataset trước khi chia
    random.shuffle(dataset)
    
    # Chia dataset thành các phần train/validation/test
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    
    return train_set, test_set

def resize_list_image(list):
    new_list = []

    for img in list:
        resized_img = cv2.resize(img, (224, 224))
        new_list.append(resized_img)

    return new_list
