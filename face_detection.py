import cv2


def detect(img):
    face_cascade = cv2.CascadeClassifier('.\cascades\haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


def detect_face_eye(img):
    face_cascade = cv2.CascadeClassifier('.\cascades\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('.\cascades\haarcascade_eye.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)


def detect_demo(filename):
    img = cv2.imread(filename)

    detect_face_eye(img)

    cv2.namedWindow('Face Detected !!!')
    cv2.imshow('Face Detected !!!', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    filename = 'screenshot_L.png'
    detect_demo(filename)
