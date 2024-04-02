import cv2

image_path = 'Lenna_color.png'
# image_path = 'Lenna_gray.bmp'

image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (50, 50))

# shape này là môt mang gồm 3 giá trị [rows, columns, channel]
num_channel = len(image.shape)

if num_channel == 2:
    print('Ảnh xám')
    cv2.imshow('Anh xam', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Ảnh màu')
    (R, G, B) = cv2.split(image)

    cv2.imshow('Red', R)
    cv2.imshow('Green', G)
    cv2.imshow('Blue', B)

    cv2.imshow('Anh mau', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()