import cv2
import numpy as np
import glob
import os
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


def extract_visual_features(gray_images):
# Extract SIFT features from gray images
    # Define our feature extractor (SIFT)
    extractor = cv2.SIFT_create()
    
    keypoints = []
    descriptors = []

    for img in gray_images:
        # extract keypoints and descriptors for each image
        img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
        if img_descriptors is not None:
            keypoints.append(img_keypoints)
            descriptors.append(img_descriptors)
    return keypoints, descriptors


def visualize_keypoints(bgr_image, image_keypoints):
    image = bgr_image.copy()
    image = cv2.drawKeypoints(image, image_keypoints, 0, (0, 255, 0), flags=0)
    return image


def split_dataset(dataset, train_ratio=0.7, test_ratio=0.1):
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
    # valid_size = int(total_size * valid_ratio)
    # test_size = total_size - train_size - valid_size
    test_size = total_size - train_size
    
    # Đảm bảo tỷ lệ không bị làm tròn lên
    assert train_size + test_size == total_size
    
    # Xáo trộn dataset trước khi chia
    random.shuffle(dataset)
    
    # Chia dataset thành các phần train/validation/test
    train_set = dataset[:train_size]
    # valid_set = dataset[train_size:train_size+valid_size]
    test_set = dataset[train_size:]
    
    return train_set, test_set
