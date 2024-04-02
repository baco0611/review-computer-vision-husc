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
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 2, 1)
    plt.bar(np.arange(len(histogram[0])), histogram[0], color='lightcoral', width=1)  # Sử dụng np.arange(len(histogram)) để tạo ra mảng indices
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Red Histogram')

    plt.subplot(2, 2, 2)
    plt.bar(np.arange(len(histogram[1])), histogram[1], color='lightgreen', width=1)  # Sử dụng np.arange(len(histogram)) để tạo ra mảng indices
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Green Histogram')

    plt.subplot(2, 2, 3)
    plt.bar(np.arange(len(histogram[2])), histogram[2], color='lightblue', width=1)  # Sử dụng np.arange(len(histogram)) để tạo ra mảng indices
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Blue Histogram')

    plt.subplot(2, 2, 4)
    plt.bar(np.arange(len(histogram[3])), histogram[3], color='gray', width=1)  # Sử dụng np.arange(len(histogram)) để tạo ra mảng indices
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Light Histogram')
    plt.show()


image_path = "test3.png"
I = cv2.imread(image_path)
I_gray = cv2.cvtColor(I, cv2.COLOR_BGRA2GRAY)
(R, G, B) = cv2.split(I)
histogram = (calcHistogram(R), calcHistogram(G), calcHistogram(B), calcHistogram(I_gray))

visualize(histogram)