import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def visualize_graph(histogram):
    plt.plot(np.arange(len(histogram)), histogram, color='lightblue', linewidth=1.25)  # Sử dụng np.arange(len(histogram)) để tạo ra mảng indices
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram')
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
        omega_1[i] = 1 - omega_0[i]

    return (omega_0, omega_1)

def calc_mean_level(pro_histogram, omega_0):
    mu_0 = np.empty(256)
    mu_1 = np.empty(256)
    mu_t = 1

    for i in range(256):
        mu_t += i * pro_histogram[i]
        mu_k = 1

        for j in range(i+1):
            mu_k += j * pro_histogram[j]

        mu_0[i] = mu_k / omega_0[i]

    for i in range(256):
        mu_1[i] = (mu_t-mu_0[i] * omega_0[i])/(1-omega_0[i])

    return (mu_t, mu_0, mu_1)

def calc_variances(omega_0, mu_1, mu_t):
    sigma_b = np.empty(256)

    for i in range(256):
        sigma_b[i] = (mu_t*omega_0[i] - mu_1[i])**2 / (omega_0[i] * (1 - omega_0[i]))
        print(sigma_b[i])

    return sigma_b

# Đọc ảnh
image_path = "shodou3.png"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Tính histogram (được làm trong hàm tính phân bố xác suất)
# histogram = calcHistogram(gray_image)
# visualize_histogram(histogram)

#Tính phân bố xác suất
probability_histogram = calc_probability_histogram(gray_image)
visualize_histogram(probability_histogram)

#Tính xác suất xuất hiện lớp (omega)
(omega_0, omega_1) = calc_class_probability(probability_histogram)
visualize_graph(omega_0)
# visualize_graph(omega_1)

#Tính trung bình mức xám
(mu_t, mu_0, mu_1) = calc_mean_level(probability_histogram, omega_0)
print(mu_t)
visualize_graph(mu_0)
# visualize_graph(mu_1)

#Tính phương sai giữa các lớp
sigma_b = calc_variances(omega_0, mu_1, mu_t)
visualize_graph(sigma_b)