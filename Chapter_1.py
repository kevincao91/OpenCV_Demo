import cv2
import numpy
import os

'''
img = numpy.zeros((3, 3), dtype=numpy.uint8)

print(img)

img_cvt = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

print(img_cvt)

print(img_cvt.shape)

image = cv2.imread('wn.png')
cv2.imwrite('MyPic.jpg', image)

image_gra = cv2.imread('wn.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('MyPic_2.jpg', image_gra)

print(image[300, 300])


randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = numpy.array(randomByteArray)


grayImage = flatNumpyArray.reshape(300, 400)
cv2.imwrite('RandomGray.png', grayImage)

bgrImage = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite('RandomColor.png', bgrImage)



image = cv2.imread('wn.png')
image[0, 0] = [255, 255, 255]

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image[:, :, 1] = 100

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



image = cv2.imread('wn.png')

image_roi = image[100:200, 400:500]
image[400:500, 400:500] = image_roi

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(image.shape)
print(image.size)
print(image.dtype)



videoCapture = cv2.VideoCapture('MyInputVid.avi')
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('MyOutputVid.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

success, frame = videoCapture.read()
while success:
    videoWriter.write(frame)
    success, frame = videoCapture.read()



cameraCapture = cv2.VideoCapture(0)
fps = 30
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('MyOutputVid.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

success, frame = cameraCapture.read()
numFramesRemaining = 10 * fps - 1
while success and numFramesRemaining > 0:
    videoWriter.write(frame)
    success, frame = cameraCapture.read()
    numFramesRemaining -= 1
cameraCapture.release()

'''


clicked = False


def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('MyWindow')
cv2.setMouseCallback('MyWindow', onMouse)
print('Showing camera feed. Click window or press any key to stop.')

success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow('MyWindow', frame)
    success, frame = cameraCapture.read()
cv2.destroyWindow('MyWindow')
cameraCapture.release()



