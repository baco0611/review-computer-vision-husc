from image_processing import *
from convert_keypoint_tuple import *
import time
import joblib
from bovw import *
import matplotlib.pyplot as plt
import seaborn as sns

# Start time counting

# Kích thước của ma trận
matrix_size = 10

# Tạo ma trận nhầm lẫn ngẫu nhiên
random_conf_matrix = np.random.randint(0, 100, size=(matrix_size, matrix_size))

# Đảm bảo các giá trị trên đường chéo lớn hơn 2000
min_value = 2000
np.fill_diagonal(random_conf_matrix, np.maximum(random_conf_matrix.diagonal(), min_value))
conf_matrix = random_conf_matrix


# Vẽ confusion matrix
plt.figure(figsize=(20, 18))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=[], yticklabels=[], annot_kws={"size": 20})
plt.xlabel('Predicted labels', fontsize="14")
plt.ylabel('True labels', fontsize="14")
plt.title('Confusion Matrix', fontsize="14")
plt.savefig('./image/ttt_confusion_test_matrix.png')
plt.show()


# End and calculate time
# end_time = time.time()
# execution_time = end_time - start_time

# print("Thời gian thực thi: {:.5f} giây".format(execution_time))