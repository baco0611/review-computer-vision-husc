from image_processing import *
from convert_keypoint_tuple import *
import time
import joblib

# Start time counting
start_time = time.time()

#Load files
print("Loading file ...")
dir = ".././dataset/CIFAR10/train/flip"
(flip_bgr, flip_gray) = load_images(dir)
dir = ".././dataset/CIFAR10/train/negative"
(neg_bgr, neg_gray) = load_images(dir)
dir = ".././dataset/CIFAR10/train/raw"
(raw_bgr, raw_gray) = load_images(dir)
dir = ".././dataset/CIFAR10/train/resize"
(rez_bgr, rez_gray) = load_images(dir)
dir = ".././dataset/CIFAR10/train/rotate"
(rotate_bgr, rotate_gray) = load_images(dir)

# print(len(bgr))
print(len(flip_bgr))
print(len(neg_bgr))
print(len(raw_bgr))
print(len(rez_bgr))
print(len(rotate_bgr))


#Extract SIFT features
print("Extracting feature flip ...")
(flip_keypoints, flip_description) = extract_visual_features(flip_gray)
print(len(flip_description))
print("Extracting feature neg ...")
(neg_keypoints, neg_description) = extract_visual_features(neg_gray)
print(len(neg_description))
print("Extracting feature raw ...")
(raw_keypoints, raw_description) = extract_visual_features(raw_gray)
print(len(raw_description))
print("Extracting feature rez ...")
(rez_keypoints, rez_description) = extract_visual_features(rez_gray)
print(len(rez_description))
print("Extracting feature rotate_ ...")
(rotate_keypoints, rotate_description) = extract_visual_features(rotate_gray)
print(len(rotate_description))
print("Define process description")
process_description = raw_description + neg_description + rez_description + rotate_description + flip_description
print(len(process_description))
# # print(type(keypoints))
# # print(type(keypoints[0]))

print("Saving data process ...")
joblib.dump(process_description, "./data/CIFAR10/process_description.joblib")
print("Saving data flip_ ...")
joblib.dump(flip_description, "./data/CIFAR10/flip_description.joblib")
print("Saving data neg_ ...")
joblib.dump(neg_description, "./data/CIFAR10/neg_description.joblib")
print("Saving data raw_ ...")
joblib.dump(raw_description, "./data/CIFAR10/raw_description.joblib")
print("Saving data rez_ ...")
joblib.dump(rez_description, "./data/CIFAR10/rez_description.joblib")
print("Saving data rotate_ ...")
joblib.dump(rotate_description, "./data/CIFAR10/rotate_description.joblib")

# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))