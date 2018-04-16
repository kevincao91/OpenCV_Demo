import numpy as np
import cv2
import csv


def read_images():
    X, Y = [], []
    face_data_file = '.\data\Face_Data.csv'

    with open(face_data_file, 'r') as input_file:
        reader = csv.reader(input_file)
        for i, row in enumerate(reader):
            file_path = row[0]
            people_id = row[1]
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            X.append(np.asarray(img, dtype=np.uint8))
            Y.append(int(people_id))

    return [X, Y]


def face_recognizer_create():
    [X, Y] = read_images()
    Y = np.asarray(Y, dtype=np.int32)

    model = cv2.face.EigenFaceRecognizer_create()
    model.train(X, Y)

    return model


def face_recognition(img, model, face_cascade, mirror, width):
    names = ['unknow', 'kevin', 'yiyi']

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = gray[x:x + w, y:y + h]

        try:
            roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
            params = model.predict(roi)
            index = params[0]
            confidence = int(params[1])

            if mirror:
                if params[1] <= 10000:
                    temp = [names[index], (width - x - w, y - 20), confidence]
                else:
                    temp = [names[0], (width - x - w, y - 20), confidence]
            else:
                if params[1] <= 10000:
                    temp = [names[index], (x, y - 20), confidence]
                else:
                    temp = [names[0], (x, y - 20), confidence]

            results.append(temp)

        except:
            print('ERROR')
            continue
    return results


def face_rec_Eigen():
    names = ['kevin', 'unknow', 'unknow']
    [X, Y] = read_images()
    Y = np.asarray(Y, dtype=np.int32)

    model = cv2.face.EigenFaceRecognizer_create()
    model.train(X, Y)
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('.\cascades\haarcascade_frontalface_default.xml')

    while True:
        read, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = gray[x:x + w, y:y + h]

            try:
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                index = params[0]
                confidence = params[1]
                print('Label: %s, Confidence: %.2f' % (index, confidence))
                if params[1] <= 5000:
                    cv2.putText(img, names[index], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                else:
                    cv2.putText(img, names[1], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                continue

        cv2.imshow('camera', img)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()


def face_rec_LBPH():
    names = ['kevin', 'unknow', 'unknow']
    [X, Y] = read_images()
    Y = np.asarray(Y, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X, Y)
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('.\cascades\haarcascade_frontalface_default.xml')

    while True:
        read, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = gray[x:x + w, y:y + h]

            try:
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                index = params[0]
                confidence = params[1]
                print('Label: %s, Confidence: %.2f' % (index, confidence))
                if params[1] <= 80:
                    cv2.putText(img, names[index], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                else:
                    cv2.putText(img, names[1], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                continue

        cv2.imshow('camera', img)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # face_rec_Eigen()
    # face_rec_LBPH()
    read_images()
