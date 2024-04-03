import cv2
import numpy as np
import matplotlib.pyplot as plt

def calc_NaN(arr):
    if np.isnan(arr[0]) or np.isinf(arr[0]):
        arr[0] = 0

    # Nếu có, thay thế NaN và Inf bằng 0
    for i in range(len(arr)):
        if np.isnan(arr[i]) or np.isinf(arr[i]):
            arr[i] = arr[i - 1] if not (np.isnan(arr[i - 1]) or np.isinf(arr[i - 1])) else 0

    return arr


def calcHistogram(I):
    histogram = np.empty(256)
    histogram = np.array([0 for x in histogram])
    I = I.flatten()

    for x in I:
        histogram[x]+=1

    return histogram

def visualize_histogram(histogram):
    plt.bar(np.arange(len(histogram)), histogram, color='lightblue', width=1)  # Sử dụng np.arange(len(histogram)) để tạo ra mảng indices
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()

def visualize_graph(histogram, name="Histogram"):
    plt.plot(np.arange(len(histogram)), histogram, color='lightblue', linewidth=1.25)  # Sử dụng np.arange(len(histogram)) để tạo ra mảng indices
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title(name)
    plt.show()

def visualize_thresgold(histogram, threshold):
    plt.plot(np.arange(len(histogram)), histogram, color='lightblue', linewidth=1.25)  # Sử dụng np.arange(len(histogram)) để tạo ra mảng indices
    plt.axvline(x=threshold, color='r', linestyle='--', label='Vertical Line')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Variances')
    plt.legend()
    plt.show()

def calc_probability_histogram(I):
    histogram = calcHistogram(I)
    total = np.sum(histogram)
    histogram = histogram / total
    return histogram

def calc_class_probability(pro_histogram):
    omega_0 = np.empty(256)
    omega_1 = np.empty(256)

    for i in range(256):
        omega_0[i] = 0
        for j in range(i+1):
            omega_0[i] += pro_histogram[j]
        if omega_0[i] > 1:
            omega_0[i] = 1
        omega_1[i] = 1 - omega_0[i]

    return (omega_0, omega_1)

def calc_mean_level(pro_histogram, omega_0, omega_1):
    mu_0 = np.empty(256)
    mu_1 = np.empty(256)
    mu_t = 1

    for i in range(256):
        mu_t += i * pro_histogram[i]
        mu_k = 0

        for j in range(i+1):
            mu_k += j * pro_histogram[j]

        mu_0[i] = mu_k / omega_0[i]

    for i in range(256):
        mu_k = 0

        for j in range(i+1):
            mu_k += j * pro_histogram[j]

        if omega_1[i] >= 0.01:
            mu_1[i] = (mu_t - mu_k)/(1 - omega_0[i])
        else:
            mu_1[i] = (mu_t - mu_k)/0

    return (mu_t, mu_0, mu_1)

def cacl_variances_sigma(omega_0, omega_1, mu_0, mu_1):
    sigma_b = np.empty(256)

    for i in range(256):
        sigma_b[i] = omega_0[i]*omega_1[i]*((mu_1[i]-mu_0[i])**2)

    return sigma_b

def get_binary(image, threshold):
    new = np.copy(image)

    for x in range(len(image)):
        for y in range(len(image[0])):
            if(image[x][y] <= threshold):
                new[x][y] = 0
            else:
                new[x][y] = 255
    
    return new

# Đọc ảnh
image_path = "Lenna_color.png"
image = cv2.imread(image_path)

height, width = image.shape[:2]
new_height = int(height / 2)
new_width = int(width / 2)
image = cv2.resize(image, (new_width, new_height))

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Tính histogram (được làm trong hàm tính phân bố xác suất)
# histogram = calcHistogram(gray_image)
# visualize_histogram(histogram)

#Tính phân bố xác suất
probability_histogram = calc_probability_histogram(gray_image)
# visualize_histogram(probability_histogram)

#Tính xác suất xuất hiện lớp (omega)
(omega_0, omega_1) = calc_class_probability(probability_histogram)
omega_0 = calc_NaN(omega_0)
omega_1 = calc_NaN(omega_1)
# visualize_graph(omega_0, 'Omega')

#Tính trung bình mức xám
(mu_t, mu_0, mu_1) = calc_mean_level(probability_histogram, omega_0, omega_1)
print(mu_t)
mu_0 = calc_NaN(mu_0)
mu_1 = calc_NaN(mu_1)
# visualize_graph(mu_0, 'Mu')

#Tính phương sai giữa các lớp
sigmaB = cacl_variances_sigma(omega_0, omega_1, mu_0, mu_1)
sigmaB = calc_NaN(sigmaB)
# visualize_graph(sigmaB, 'Sigma')

threshold = np.argmax(sigmaB)
# visualize_thresgold(sigmaB, threshold)

binary_image = get_binary(gray_image, threshold)
cv2.imshow('Image', image)
cv2.imshow('Gray image', gray_image)
cv2.imshow('Binary image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()