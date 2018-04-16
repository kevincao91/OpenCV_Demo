import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

'''
img = cv2.imread('qp.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 23, 0.04)
img[dst > 0.01 * dst.max()] = [0, 0, 255]
while True:
    cv2.imshow('corners', img)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
'''

'''
img = cv2.imread('qp.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def fd(algorithm):
    if algorithm == 'SIFT':
        return cv2.xfeatures2d.SIFT_create()
    if algorithm == 'SURF':
        return cv2.xfeatures2d.SURF_create(8000)


fd_alg = fd('SURF')
keypoints, descriptor = fd_alg.detectAndCompute(gray, None)

img = cv2.drawKeypoints(image=img, keypoints=keypoints, outImage=img, color=(51, 163, 236),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
while True:
    cv2.imshow('sift_keypoints', img)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
'''

'''
img1 = cv2.imread('music_logo.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('music_pic.png', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:40], img2, flags=2)
plt.imshow(img3), plt.show()
'''


'''
img1 = cv2.imread('screenshot_L.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('screenshot_R.png', cv2.IMREAD_GRAYSCALE)

# create SIFT and detect/compute
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# FLANN matcher parameters
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(indexParams, searchParams)

matches = flann.knnMatch(des1, des2, k=2)

# prepare an empty mask to draw good matches
matchesMask = [[0, 0] for i in range(len(matches))]

# David G. Lowe's ratio test, populate the mask
for i, (m, n) in enumerate(matches):
    if m.distance < 0.6*n.distance:
        matchesMask[i] = [1, 0]

drawParams = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)

resultImage = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **drawParams)

plt.imshow(resultImage), plt.show()
'''


MIN_MATCH_COUNT = 10
img1 = cv2.imread('music_logo.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('music_pic2.png', cv2.IMREAD_GRAYSCALE)

# create SIFT and detect/compute
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN matcher parameters
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(indexParams, searchParams)

matches = flann.knnMatch(des1, des2, k=2)

# prepare an empty mask to draw good matches
matchesMask = [[0, 0] for i in range(len(matches))]

# David G. Lowe's ratio test, populate the mask
for i, (m, n) in enumerate(matches):
    if m.distance < 0.6*n.distance:
        matchesMask[i] = [1, 0]

drawParams = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)

resultImage = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **drawParams)

plt.imshow(resultImage), plt.show()



