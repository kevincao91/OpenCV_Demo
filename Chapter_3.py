import cv2
import numpy as np
from scipy import ndimage


kernel_3_3 = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

kernel_5_5 = np.array([[-1, -1, -1, -1, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, 2, 4, 2, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, -1, -1, -1, -1]])

img = cv2.imread('screenshot.png', 0)

k3 = ndimage.convolve(img, kernel_3_3)
k5 = ndimage.convolve(img, kernel_5_5)

blurred = cv2.GaussianBlur(img, (11, 11), 0)
g_hpf = img - blurred

cv2.imshow('ori', img)
cv2.imshow('3_3', k3)
cv2.imshow('5_5', k5)
cv2.imshow('g_hpf', g_hpf)
cv2.waitKey()
cv2.destroyAllWindows()

