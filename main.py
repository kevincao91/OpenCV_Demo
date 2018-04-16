import cv2


count = 0
print(count)

img = cv2.imread('dog_cat.jpeg')
cv2.imshow('new', img)
cv2.waitKey()
cv2.destroyAllWindows()


help(cv2.face)

