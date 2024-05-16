from bovw import *
import joblib
import time

# Start time counting
start_time = time.time()


print("Loading data ...")
size = 200
name = "process"
date = "20240514"
description = joblib.load(f'./data/{name}_description.joblib')
all_descriptor = description

print("Training codebook ...")
codebook = build_codebook(all_descriptor, size)
# print(codebook)

print("Save codebook ...")
path_codebook = f'./data/{date}_{name}_{size}_codebook.joblib'
save_codebook(codebook, path_codebook)

# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))