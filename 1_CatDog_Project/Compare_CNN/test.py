import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import plot_model

def VGG11(input_shape=(224, 224, 3), num_classes=1000):
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.2),
        Dense(4096, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create the VGG11 model
model = VGG11()


def plot_keras_model(model, show_shapes=True, show_layer_names=True, layer_spacing=0.1):
    num_layers = len(model.layers)
    plt.figure(figsize=(10, num_layers * layer_spacing))
    
    # Tạo ra một từ điển để lưu trữ tên và hình dạng của các lớp
    nodes = {}
    for i, layer in enumerate(model.layers):
        nodes[i] = (layer.name, layer.output_shape)

    # Vẽ các lớp và kết nối
    for i in range(num_layers):
        name, shape = nodes[i]
        plt.text(0.5, num_layers - i - 1, f'{name}\n{shape}', ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue'))
        if i < num_layers - 1:
            plt.arrow(0.5, num_layers - i - 1.5, 0, -0.4, head_width=0.1, head_length=0.1, fc='k', ec='k')

    # Định dạng đồ thị
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Chỉnh sửa giá trị 'layer_spacing' để thay đổi khoảng cách giữa các lớp
plot_keras_model(model, layer_spacing=1.5)