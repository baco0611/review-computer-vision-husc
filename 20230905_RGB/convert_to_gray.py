import cv2

image = cv2.imread('Lenna_color.png')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('Lenna_gray.bmp', gray_image)

cv2.imshow('Image', image)
cv2.imshow('Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()