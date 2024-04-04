from bovw import *
import joblib
import time

# Start time counting
start_time = time.time()


print("Loading data ...")
cat_description = joblib.load('./data/cat_description.joblib')
dog_description = joblib.load('./data/dog_description.joblib')
all_descriptor = cat_description + dog_description

print("Training codebook ...")
codebook = build_codebook(all_descriptor)
# print(codebook)

print("Save codebook ...")
path_codebook = './data/codebook.joblib'
save_codebook(codebook, path_codebook)

# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))