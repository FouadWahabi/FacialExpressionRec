import time

import cv2

from numpysocket import *

cascade_classifier = cv2.CascadeClassifier("./haarcascade_files/haarcascade_frontalface_default.xml")


def brighten(data, b):
    datab = data * b
    return datab


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None
        # cv2.imshow("Lol", image)
        # cv2.waitKey(0)

    return image


nps = numpysocket()

video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# while True:
# Capture frame-by-frame
ret, frame = video_capture.read()

image = format_image(frame)

# Display face
cv2.imshow("Lol", image)

# Display the resulting frame
cv2.imshow('Video', frame)

nps.startClient("www.math-cs.ucmo.edu", image)

if cv2.waitKey(1) & 0xFF == ord('q'):
    pass

time.sleep(0.1)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
