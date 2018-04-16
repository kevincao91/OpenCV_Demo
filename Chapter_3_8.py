import cv2
import numpy as np

'''
img = np.zeros((200, 200), dtype=np.uint8)
img[50:150, 50:150] = 255

ret, thresh = cv2.threshold(img, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)
cv2.imshow('contours', color)
cv2.waitKey()
cv2.destroyAllWindows()

'''

img = cv2.pyrDown(cv2.imread('wn.png', cv2.IMREAD_UNCHANGED))

ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

'''
for c in contours:
    # find bounding box coordinates
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 3)

    # find minimum area
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)  # calculate coordinates of the minimum area rectangle
    box = np.int0(box)  # normalize coordinates to integers
    cv2.drawContours(img, [box], -1, (0, 255, 255), 2)  # draw contours

    # calculate center and radius of minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))  # cast to integers
    radius = int(radius)
    img = cv2.circle(img, center, radius, (255, 0, 255), 2)  # draw the circle

cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
cv2.imshow('contours', img)
cv2.waitKey()
cv2.destroyAllWindows()

'''

for cnt in contours:
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    hull = cv2.convexHull(cnt)
    cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
    cv2.drawContours(img, [hull], -1, (255, 0, 0), 2)

cv2.imshow("hull", img)
cv2.waitKey()
cv2.destroyAllWindows()
