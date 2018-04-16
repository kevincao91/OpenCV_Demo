import cv2
from managers import WindowManager, CaptureManager
import filters
import face_detection as fd
import functions as fn
import time


class Cameo(object):

    def __init__(self):
        self._isMirror = True
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, self._isMirror)
        self._curveFilter = filters.BGRPortraCurveFilter()
        self._face_cascade = cv2.CascadeClassifier('.\cascades\haarcascade_frontalface_default.xml')
        self._model = fn.face_recognizer_create()
        self._time1 = time.time()
        self._time2 = self._time1
        self.count = 0

    def run(self):
        # run the main loop
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            # TODO: Filter the frame.
            '''
            # draw to the window, if any.
            if self.previewWindowManager is not None:
                if self.shouldMirrorPreview:
                    mirroredFrame = numpy.fliplr(self._frame).copy()
                    self.previewWindowManager.show(mirroredFrame)
                else:
                    self.previewWindowManager.show(self._frame)
            '''

            # fd.detect(frame)
            results = fn.face_recognition(frame, self._model, self._face_cascade, self._isMirror, self._captureManager.width)

            # filters.strokeEdges(frame, frame)
            # self._curveFilter.apply(frame, frame)

            self._captureManager.exitFrame(results)
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        ''' handle a keypress.

        space ->   take a screenshot
        tab   ->   start/stop recording a screencast
        escape->   quit.

        '''

        if keycode == 32:  # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9:  # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27:  # escape
            self._windowManager.destroyWindow()


if __name__ == '__main__':
    Cameo().run()
