import joblib
import time
from Processing_function import load_images

# Start time counting
start_time = time.time()

print("Loading data ...")
cat_dir = "../image/Cat1000"
cat_neg_dir = "../image/CatNeg"
cat_resize_dir = "../image/CatResize"
cat_rotate_dir = "../image/CatRotate"
(cat_bgr, _) = load_images(cat_dir)
(cat_neg_bgr, _) = load_images(cat_neg_dir)
(cat_resize_bgr, _) = load_images(cat_resize_dir)
(cat_rotate_bgr, _) = load_images(cat_rotate_dir)

dog_dir = "../image/Dog1000"
dog_neg_dir = "../image/DogNeg"
dog_resize_dir = "../image/DogResize"
dog_rotate_dir = "../image/DogRotate"
(dog_bgr, _) = load_images(dog_dir)
(dog_neg_bgr, _) = load_images(dog_neg_dir)
(dog_resize_bgr, _) = load_images(dog_resize_dir)
(dog_rotate_bgr, _) = load_images(dog_rotate_dir)

cat_process_dir = "../Cat_process"
dog_process_dir = "../Dog_process"
(dog_process, _ ) = load_images(dog_process_dir)
(cat_process, _ ) = load_images(cat_process_dir)

print("Saving data ...")
joblib.dump(cat_bgr, "./data/cat_regular.joblib")
joblib.dump(cat_neg_bgr, "./data/cat_neg.joblib")
joblib.dump(cat_resize_bgr, "./data/cat_resize.joblib")
joblib.dump(cat_rotate_bgr, "./data/cat_rotate.joblib")
joblib.dump(cat_process, "./data/cat_process.joblib")

joblib.dump(dog_bgr, "./data/dog_regular.joblib")
joblib.dump(dog_neg_bgr, "./data/dog_neg.joblib")
joblib.dump(dog_resize_bgr, "./data/dog_resize.joblib")
joblib.dump(dog_rotate_bgr, "./data/dog_rotate.joblib")
joblib.dump(dog_process, "./data/dog_process.joblib")

# End and calculate time
end_time = time.time()
execution_time = end_time - start_time
print("Thời gian thực thi: {:.5f} giây".format(execution_time))
