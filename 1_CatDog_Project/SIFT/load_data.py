from image_processing import *
from convert_keypoint_tuple import *
import time
import joblib

# Start time counting
start_time = time.time()

#Load files
print("Loading file ...")
cat_dir = "../Cat1000"
cat_dir = "../Cat_process"
(cat_bgr, cat_gray) = load_images(cat_dir)
dog_dir = "../Dog1000"
dog_dir = "../Dog_process"
(dog_bgr, dog_gray) = load_images(dog_dir)


#Extract SIFT features
print("Extracting feature ...")
(cat_keypoints, cat_description) = extract_visual_features(cat_gray)
(dog_keypoints, dog_description) = extract_visual_features(dog_gray)

print(type(cat_keypoints))
print(type(cat_keypoints[0]))

print("Saving data ...")
tuple_keypoints = keypoints_to_data(cat_keypoints)
joblib.dump(tuple_keypoints, "./data/cat_keypoints.joblib")
joblib.dump(cat_description, "./data/cat_description.joblib")

tuple_keypoints = keypoints_to_data(dog_keypoints)
joblib.dump(tuple_keypoints, "./data/dog_keypoints.joblib")
joblib.dump(dog_description, "./data/dog_description.joblib")

# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))