import cv2
import numpy as np
import matplotlib.pyplot as plt

def calcHistogram(I):
    histogram = np.empty(256)
    histogram = [0 for x in histogram]
    I = I.flatten()

    for x in I:
        histogram[x]+=1

    return histogram

def visualize(histogram):
    plt.bar(np.arange(len(histogram)), histogram, color='lightblue', width=1)  # Sử dụng np.arange(len(histogram)) để tạo ra mảng indices
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()

def calc_threshold(histogram, epsilon):
    threshold = 100
    pre = 0

    while threshold - pre > epsilon:
        under_threshold = [0, 0]
        over_threshold = [0, 0]
        pre = threshold
        threshold = int(threshold)

        for i in range(threshold + 1):
            under_threshold[0] += histogram[i]
            under_threshold[1] += histogram[i] * i

        for i in range(threshold + 1, len(histogram)):
            over_threshold[0] += histogram[i]
            over_threshold[1] += histogram[i] * i

        under_threshold[1]/=under_threshold[0]
        over_threshold[1]/=over_threshold[0]
        threshold = (under_threshold[1] + over_threshold[1])/2

    return int(threshold)

def bimodal_segmentation(image, epsilon=0.5):
    T = np.random.randint(0, 256)
    
    while True:
        G1 = image[image <= T]
        G2 = image[image > T]
        
        m1 = np.mean(G1)
        m2 = np.mean(G2)
        
        new_T = (m1 + m2) / 2
        
        if abs(T - new_T) < epsilon:
            break
        
        T = new_T
    
    return T

def binary_image(I, threshold):
    new_image = I.copy()

    for x in range(len(I)):
        for y in range(len(I[0])):
            if I[x][y] <= threshold:
                new_image[x][y] = 0
            else:
                new_image[x][y] = 255

    return new_image 

image_path = "highcontrast1.png"
image = cv2.imread(image_path)
(B, G, R) = cv2.split(image)
r_histogram = calcHistogram(R)
g_histogram = calcHistogram(G)
b_histogram = calcHistogram(B)

epsilon = 0.0001
r_threshold = calc_threshold(r_histogram, epsilon)
g_threshold = calc_threshold(g_histogram, epsilon)
b_threshold = calc_threshold(b_histogram, epsilon)

image_r = binary_image(R, r_threshold)
image_g = binary_image(G, g_threshold)
image_b = binary_image(B, b_threshold)

rgb_image = np.stack([image_b, image_g, image_r], axis=2).astype(np.uint8)

# Hiển thị ảnh bằng OpenCV
cv2.imshow('RGB Image', rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()