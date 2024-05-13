from bovw import *
import joblib
import time

# Start time counting
start_time = time.time()


print("Loading data ...")

data_flow = "process"
all_descriptor = joblib.load(f'./data/CIFAR10/{data_flow}_description.joblib')

date = "20240510"
size = 20
print("Training codebook ...")
codebook = build_codebook(all_descriptor, size)
# print(codebook)

name_of_codebook = f"{date}_{data_flow}_{size}"
print("Save codebook ...")
path_codebook = f'./data/{name_of_codebook}_codebook.joblib'
save_codebook(codebook, path_codebook)

# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))