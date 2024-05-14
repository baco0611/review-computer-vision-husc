from tensorflow.keras.datasets import cifar10
import joblib

data = cifar10.load_data()
joblib.dump(data, "../dataset/CIFAR10/raw_data.joblib")
