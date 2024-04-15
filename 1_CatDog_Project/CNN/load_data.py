import time
import joblib
from Processing_function import load_images


# Start time counting
start_time = time.time()

print("Loading data ...")
#Load files
cat_dir = "../Cat1000"
cat_dir = "../Cat_process"
cat_dir = "../Cat_process_nonNeg"
(cat_bgr, cat_gray) = load_images(cat_dir)
dog_dir = "../Dog1000"
dog_dir = "../Dog_process"
dog_dir = "../Dog_process_nonNeg"
(dog_bgr, dog_gray) = load_images(dog_dir)

print("Saving data ...")
joblib.dump(cat_bgr, "./data/cat_bgr_nonNeg.joblib")
joblib.dump(cat_gray, "./data/cat_gray_nonNeg.joblib")
joblib.dump(dog_bgr, "./data/dog_bgr_nonNeg.joblib")
joblib.dump(dog_gray, "./data/dog_gray_nonNeg.joblib")

# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))