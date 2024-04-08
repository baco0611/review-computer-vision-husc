from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
from numpy import expand_dims
import joblib
import cv2


print("Loading image ...")
cat_bgr = joblib.load("./data/cat_bgr.joblib")
dog_bgr = joblib.load("./data/dog_bgr.joblib")
image = dog_bgr[200]
image = cv2.resize(image, (224, 224))

print("Loading model ...")
model_name = "VGG11"
model = load_model("./data/" + model_name + "_CNN_model.h5")
print(model.summary())

print("\nModel's layer ...")
# for layer in model.layers:
#     if 'conv' not in layer.name:
#         continue    
#     filters , bias = layer.get_weights()
#     print(layer.name , filters.shape)

# filters , bias = model.layers[0].get_weights()

# # normalize filter values to 0-1 so we can visualize them
# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min) / (f_max - f_min)

# n_filters = 16
# ix=1
# fig = plt.figure(figsize=(20,15))
# for i in range(n_filters):
#     # get the filters
#     f = filters[:,:,:,i]
#     for j in range(3):
#         # subplot for 6 filters and 3 channels
#         plt.subplot(n_filters,3,ix)
#         plt.imshow(f[:,:,j] ,cmap='gray')
#         ix+=1
# #plot the filters 
# plt.show()

blocks = []

for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:
        continue    
    print(i , layer.name , layer.output.shape)
    blocks.append(i)

outputs = [model.layers[i].output for i in blocks]

model = Model(inputs=model.inputs , outputs=outputs)
# convert the image to an array
image = img_to_array(image)
# expand dimensions so that it represents a single 'sample'
image = expand_dims(image, axis=0)

features_map = model.predict(image)

for i, fmap in zip(blocks, features_map):
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("BLOCK_{}".format(i), fontsize=20)
    num_maps = fmap.shape[3]
    max_cols = 8
    max_rows = (num_maps + max_cols - 1) // max_cols  # Số hàng được giới hạn theo số cột tối đa

    for j in range(1, max_rows * max_cols + 1):
        if j <= num_maps:
            plt.subplot(max_rows, max_cols, j)
            if j <= fmap.shape[3]:
                plt.imshow(fmap[0, :, :, j - 1])
                plt.title("Map {}".format(j - 1))  # Hiển thị chỉ số của feature map
            else:
                plt.axis('off')  # Tắt trục nếu không có đủ feature map để điền vào ô trống
        else:
            plt.axis('off')  # Tắt trục nếu không có đủ feature map để điền vào ô trống
    
    # Lưu hình ảnh của block vào file
    block_filename = "./image/{}_block{}.png".format(model_name, i)
    plt.savefig(block_filename)
    plt.close(fig)


plt.show()
    