import imutils
import numpy
import time
import cv2
from PIL import Image
import tf_runner as tf

camera = cv2.VideoCapture(0)
time.sleep(1)

firstFrame = None

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
zeros = None
width = 512
height = 384
MAX_WIDTH = 72
MAX_HEIGHT = 170

take_photo = False
is_started = False

tf.init()

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized


while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
        break
    frame = imutils.resize(frame, width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = blurred_gray
        continue

    frameDelta = cv2.absdiff(firstFrame, blurred_gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, contourArray, _) = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for index, contour in enumerate(contourArray):

        if cv2.contourArea(contour) < 500:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(blurred_gray, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if is_started:
            cropped_image = Image.fromarray(gray[x:x+w, y:y+h])
            #wpercent = (MAX_WIDTH / float(w))
            #hsize = int((float(w) * float(wpercent)))
            resized_image = cropped_image.resize((MAX_WIDTH, MAX_HEIGHT), Image.ANTIALIAS)
            resized_image.save("out{}.png".format(index))
            tf.test_frame(resized_image)

    cv2.imshow("Security Feed", frame)
    cv2.imshow("Frame Delta", blurred_gray)
    cv2.imshow("Thresh", thresh)

    FRAMEDELTA = cv2.merge([frameDelta, frameDelta, frameDelta])
    THRESH = cv2.merge([thresh, thresh, thresh])
    GRAY = cv2.merge([blurred_gray, blurred_gray, blurred_gray])

    output = numpy.zeros((height * 2, width * 2, 3), dtype="uint8")
    output[0:height, 0:width] = GRAY
    output[0:height, width:width * 2] = FRAMEDELTA
    output[height:height * 2, 0:width] = THRESH
    output[height:height * 2, width:width * 2] = frame


    ##------------------------------------------------------------------------------
    ##  If the `q` key is pressed then exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord('r'):
        firstFrame = blurred_gray
        is_started = True;
    elif key == ord('p'):
        take_photo = True



##------------------------------------------------------------------------------
##  Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()


