import cv2
import numpy as np
from numba import jit


def generateASCIILetters():
    images = []
    letters = "# $%&\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    #letters = " \\ '(),-./:;[]_`{|}~"
    for letter in letters:
        img = np.zeros((12, 16), np.uint8)
        img = cv2.putText(img, letter, (0, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        images.append(img)

    return np.stack(images)


@jit(nopython = True)
def toASCIIArt(frame_temp, images_temp, box_height = 12, box_width = 16):
    height, width = frame_temp.shape

    for i in range(0, height, box_height):
        for j in range(0, width, box_width):
            roi = frame_temp[i:i + box_height, j:j + box_width]
            best_match = np.inf
            best_match_index = 0

            for k in range(1, images_temp.shape[0]):
                total_sum = np.sum(np.absolute(np.subtract(roi, images_temp[k])))

                if total_sum < best_match:
                    best_match = total_sum
                    best_match_index = k

            roi[:, :] = images_temp[best_match_index]

    return frame_temp


vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

images = generateASCIILetters()

if vc.isOpened() :
    rval, frame = vc.read()
else :
    rval = False

while rval :
    rval, frame = vc.read()
    frame = cv2.flip(frame, 1)

    gb = cv2.GaussianBlur(frame, (5, 5), 0)
    can = cv2.Canny(gb, 127, 31)

    cv2.imshow("Can", can)
    cv2.imshow("Canny edge detection", toASCIIArt(can, images))
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == 27 :
        break

vc.release()
cv2.destroyAllWindows()
