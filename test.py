import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print('GPU được sử dụng:')
    for gpu in gpus:
        print(gpu)
else:
    print('Không tìm thấy GPU. Sử dụng CPU.')