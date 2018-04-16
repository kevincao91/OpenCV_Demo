import cv2
import numpy
import time


class CaptureManager(object):

    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):

        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview

        self._capture = capture
        self._chanel = 0
        self._enteredFrame = False
        self._frame = None
        self._mirroredFrame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._framesElapsed = numpy.long(0)
        self._fpsEstimate = None

        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def channel(self):
        return self._chanel

    @channel.setter
    def channel(self, value):
        if self._chanel != value:
            self._chanel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def enterFrame(self):
        # capture the next frame, if any.
        # but first, check that any previous frame was exited.
        assert not self._enteredFrame, \
            'previous enterFrame() had no matching exitFrame()'

        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self, results):
        # draw to the window. write to files. release the frame.
        # check whether any grabbed frame is retrievable.
        # the getter may retrieve and cache the frame.
        if self.frame is None:
            self._enteredFrame = False
            return

        # update the FPS estimate and related variables.
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / timeElapsed
            # show fps
            # print(self._fpsEstimate)
        self._framesElapsed += 1

        # draw to the window, if any.
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                self._mirroredFrame = numpy.fliplr(self._frame).copy()
                for result in results:
                    cv2.putText(self._mirroredFrame, result[0] + str(result[2]), result[1], cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                self.previewWindowManager.show(self._mirroredFrame)
            else:
                for result in results:
                    cv2.putText(self._frame, result[0] + str(result[2]), result[1], cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                self.previewWindowManager.show(self._frame)

        # write to the image file, if any.
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None

        # write to the video file, if any.
        self._writingVideoFrame()

        # release the frame.
        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        # write the next exited frame to an image file.
        self._imageFilename = filename

    def startWritingVideo(self, filename, encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        # start writing exited frames to a video file.
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        # stop writing exited frames to a video file.
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writingVideoFrame(self):
        if not self.isWritingVideo:
            return
        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                # the capture's FPS is unknown so use an estimate
                if self._framesElapsed < 20:
                    # write until more frames elapse so that the estimate is more stable.
                    return
                else:
                    fps = self._fpsEstimate
            size = (self.width, self.height)
            self._videoWriter = cv2.VideoWriter(self._videoFilename, self._videoEncoding, fps, size)
        if self.shouldMirrorPreview:
            self._videoWriter.write(self._mirroredFrame)
        else:
            self._videoWriter.write(self._frame)


class WindowManager(object):

    def __init__(self, windowName, keypressCallback=None):
        self.keypressCallback = keypressCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            # discard any non-ASCII info encoded by GTK.
            keycode &= 0xFF
            self.keypressCallback(keycode)
