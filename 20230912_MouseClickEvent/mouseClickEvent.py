import cv2

def onClick(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Kiểm tra nếu nhấp chuột trái
            # Lấy giá trị của pixel tại tọa độ (x, y)
            pixel_value = image[y, x]
            print(f"Pixel value at position ({x}, {y}): {pixel_value}")

image_path = 'Lenna_color.png'
image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.resize(image, (1024, 1024))
cv2.imshow('Image', image)
cv2.setMouseCallback('Image', onClick)

cv2.waitKey(0)
cv2.destroyAllWindows()