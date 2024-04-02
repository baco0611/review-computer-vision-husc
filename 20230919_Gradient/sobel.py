import cv2
import numpy as np

def calc_gradient(I, point=(1, 1)):
    x, y = point
    # Sửa lỗi truy cập ngoài giới hạn bằng cách padding
    padded_I = np.pad(I, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    x += 1
    y += 1

    Gx = padded_I[x, y+1] - padded_I[x, y-1]
    Gy = padded_I[x-1, y] - padded_I[x+1, y]

    magnitude = round(np.sqrt(Gy**2 + Gx**2), 1)
    orientation = round(np.arctan2(Gy, Gx), 1)
    
    return orientation, magnitude

def calc_gradient_image(I):
    gradient_orientation = np.zeros_like(I, dtype=float)
    gradient_magnitude = np.zeros_like(I, dtype=float)

    for x in range(I.shape[0]):
        for y in range(I.shape[1]):
            orientation, magnitude = calc_gradient(I, (x, y))
            gradient_orientation[x, y] = orientation
            gradient_magnitude[x, y] = magnitude
    
    return gradient_orientation, gradient_magnitude

# Sử dụng cv2.filter2D() để tính gradient
def gradient_cv2(I):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Gx = cv2.filter2D(I, -1, sobel_x)
    Gy = cv2.filter2D(I, -1, sobel_y)

    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx) * (180 / np.pi)
    
    return orientation, magnitude

# Ví dụ sử dụng:
# Đảm bảo I là một ảnh xám được nạp thông qua cv2.imread() và cv2.cvtColor() nếu cần
I = cv2.imread('test3.png', cv2.IMREAD_GRAYSCALE)
I = cv2.resize(I, (10, 10))

print(I)
print()

# Áp dụng calc_gradient_image
gradient_orientation, gradient_magnitude = calc_gradient_image(I)

print(gradient_orientation)
print()
print(gradient_magnitude)
print()

# Áp dụng gradient_cv2
orientation_cv2, magnitude_cv2 = gradient_cv2(I)
print(orientation_cv2)
print()
print(magnitude_cv2)
print()