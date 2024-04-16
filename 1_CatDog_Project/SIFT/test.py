from image_processing import *
from convert_keypoint_tuple import *
import time
import joblib
from bovw import *

# Start time counting
start_time = time.time()

codebook = joblib.load("./data/codebook.joblib")

#Load files
print("Loading file ...")
cat_dir = "../Cat1000"
(cat_bgr, cat_gray) = load_images(cat_dir)


#Extract SIFT features
print("Extracting feature ...")
(cat_keypoints, cat_description) = extract_visual_features(cat_gray)

img = cat_bgr[1]
img_with_keypoints = cv2.drawKeypoints(cat_bgr[1], cat_keypoints[1], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
output_file_name = "cat_with_keypoints.jpg"
cv2.imwrite(output_file_name, img_with_keypoints)
feature = represent_image_features(cat_description[1], codebook)
print(feature)


# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))