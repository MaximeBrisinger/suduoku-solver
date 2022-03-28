import os
import cv2
import numpy as np


def display(predictions):

    folder = "data/digits_to_display/"
    w, h = 100, 100

    grid = np.zeros((h * 9, w * 9))

    dict_digits = dict()
    for file in os.listdir(folder):
        img = cv2.imread(folder + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_np = np.asarray(img)

        digit = int(file.replace(".jpg", ""))
        dict_digits[digit] = img_np

    for k in range(len(predictions)):
        i = int(k / 9)
        j = k % 9
        value = predictions[k]

        # Plot digits
        grid[i * w: (i + 1) * w, j * h: (j + 1) * h] = dict_digits[value]

        # Draw lines
        # grid[i * w: i * w + 3, j * h: (j + 1) * h] = 0
        # grid[i * w: (i + 1) * w, j * h: j * h + 3] = 0

    # Draw lines
    for k in range(9):
        grid[k * w: k * w + 3, :] = 0
        grid[:, k * h: k * h + 3] = 0

    grid[-3: -1, :] = 0
    grid[:, - 3: -1] = 0

    # Draw main lines
    for k in range(1, 3):
        grid[3 * k * w: 3 * k * w + 10, :] = 0
        grid[:, 3 * k * h: 3 * k * h + 10] = 0

    grid = 255 - grid
    return grid


if __name__ == '__main__':
    predicted = [8, 2, -1, 9, -1, -1, 2, -1, 8, -1, -1, -1, -1, -1, 2, 7, 8, -1, -1, 6, -1, 8, -1, -1, 9, 6, -1, -1, -1, -1, -1, -1, 8, 8, -1, 1, 9, 6, -1, 8, -1, 7, -1, 2, 8, 7, -1, 8, 6, -1,
                 -1, -1, -1, -1, -1, 8, 8, -1, -1, 1, -1, 9, -1, -1, 7, 1, 3, -1, -1, -1, -1, -1, 2, -1, 6, -1, -1, 2, -1, 6, 7]
    # 15 / 36 bien detectes

    pred2 = [6, 8, -1, -1, 7, -1, -1, -1, -1, 6, -1, -1, 4, 9, 5, -1, -1, -1, -1, 9, 8, -1, -1, -1, -1, 6, -1, 8, -1, -1, -1, 6, -1, -1, -1, 8, 1, -1, -1, 2, -1, 8, -1, -1, 4, 7, -1, -1, -1,
             2, -1, -1, -1, 6, -1, 6, -1, -1, -1, -1, 2, 8, -1, -1, -1, -1, 3, 4, 9, -1, -1, 5, -1, -1, -1, -1, 8, -1, -1, 7, 9]
    # 20 / 30 bien detectes


    display(predicted)
