import cv2
import numpy as np
from numba import jit
import math

letters = "# $%&\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~"
letters_length = len(letters)


def generateASCIILetters():
    images = []
    # letters = " \\ '(),-./:;[]_`{|}~"

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


def main() :
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    images = generateASCIILetters()

    if vc.isOpened() :
        rval, frame = vc.read()
    else :
        rval = False

    frame_number = -1

    while rval :
        frame_number += 1
        rval, frame = vc.read()
        frame = cv2.flip(frame, 1)

        if frame_number == 50:
            height = frame.shape[0]
            width = frame.shape[1]

            string = ""

            for i in range(height):
                for j in range(width):
                    string += letters[int(frame[i, j][0] / 255 * letters_length) - 1]

                string += '\n'

            print(string)

            frame_number = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Webcam", gray)

        if cv2.waitKey(1) == 27 :
            break


if __name__ == '__main__' :
    main()
