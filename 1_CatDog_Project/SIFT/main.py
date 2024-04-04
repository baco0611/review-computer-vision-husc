from image_processing import *

cat_dir = "../Cat1000"
(cat_bgr, cat_gray) = load_images(cat_dir)
dog_dir = "../Dog1000"
(dog_bgr, dog_gray) = load_images(dog_dir)