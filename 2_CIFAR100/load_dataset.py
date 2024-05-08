import pickle
import cv2
import time
import numpy as np

# Start time counting
start_time = time.time()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


print("Load dataset ...")
direct = "train"
dict = unpickle(f"./cifar-100-python/{direct}")
print(dict.keys())

file_name = dict[b'filenames']
batch_label = dict[b'batch_label']
fine_label = dict[b'fine_labels']
coarse_label = dict[b'coarse_labels']
data = dict[b'data']

print("Saving label ...")
fine_label = np.array(fine_label)
np.savez("./dataset/train/label.npz", fine_label)

print("Process data ...")
data = np.array(data)
data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

cv2.imshow('image', data[0])
cv2.waitKey(3000)
cv2.destroyAllWindows()

print("Saving data ...")
i = 0
for x in data:
    cv2.imwrite(f"./dataset/{direct}/raw/{i}.png", x)
    i+=1

# End and calculate time
end_time = time.time()
execution_time = end_time - start_time
print("Thời gian thực thi: {:.5f} giây".format(execution_time))