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


image_path = "highcontrast1.png"
I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
histogram = calcHistogram(I)

visualize(histogram)