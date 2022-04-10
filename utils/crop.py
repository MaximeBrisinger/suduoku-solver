import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(folder, file_name, display=False):
    sudoku_a = cv2.imread(folder + '/' + file_name)

    if display:
        plt.figure()
        plt.imshow(sudoku_a)

    # Preprocessing image to be read
    sudoku_a = cv2.resize(sudoku_a, (450, 450))

    return sudoku_a


def preprocess(image, display=False):
    """
        Blurs and thresholds the image to get a binary one
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 6)
    threshold_img = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    if display:
        plt.figure()
        plt.imshow(threshold_img)
    return threshold_img


def main_outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new


def splitcells(img, display=False):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    if display:
        plt.figure()
        plt.imshow(boxes[80])
    return boxes

