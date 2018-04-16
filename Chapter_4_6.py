import numpy as np
import cv2
import matplotlib
from matplotlib import  pyplot as plt

img = cv2.imread('screenshot_L.png')
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (200, 25, 250, 375)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img2 = img * mask2[:, :, np.newaxis]

cv2.rectangle(img, (200, 25), (450, 400), (255, 0, 0), 4)

plt.subplot(121), plt.imshow(img2)
plt.title('grabcut'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread('screenshot_L.png'), cv2.COLOR_BGR2GRAY))
plt.subplot(122), plt.imshow(img)
plt.title('original'), plt.xticks([]), plt.yticks([])
plt.show()



