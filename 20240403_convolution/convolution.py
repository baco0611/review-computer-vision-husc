import cv2
import numpy as np
import os

def apply_kernel(image, kernel, kernel_name):
    # Áp dụng phép tính convolution với kernel đã cho
    result = cv2.filter2D(image, -1, kernel)
    # Lưu ảnh kết quả với tên tương ứng của kernel
    cv2.imwrite(os.path.join("image", f"{kernel_name}.jpg"), result)

# Load ảnh
image_path = "jisoo.jpg"
image = cv2.imread(image_path)

# Resize ảnh về kích thước 512x512
image_resized = cv2.resize(image, (512, 512))

# Lưu ảnh gốc
cv2.imwrite("./image/original_image.jpg", image_resized)

# Tạo thư mục "image" nếu chưa tồn tại
os.makedirs("image", exist_ok=True)

# Các kernel được cung cấp
kernels = [
    (np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), "sobel_x"),
    (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), "sobel_y"),
    (np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]), "prewitt_x"),
    (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), "prewitt_y"),
    (np.array([[1, 0], [0, -1]]), "roberts"),
    (np.array([[1, 1, -1], [1, -2, -1], [1, 1, -1]]), "corner_1"),
    (np.array([[-1, 1, 1], [-1, -2, 1], [-1, 1, 1]]), "corner_2"),
    (np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16, "gaussian_3x3"),
    (np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9, "box_blur_3x3"),
    (np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]), "laplacian")
]

# Áp dụng các kernel và lưu ảnh kết quả
for kernel, kernel_name in kernels:
    apply_kernel(image_resized, kernel, kernel_name)
