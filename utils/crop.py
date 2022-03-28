import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(folder, file_name, display=False):
    print(file_name)
    sudoku_a = cv2.imread(folder + '/' + file_name)

    if display:
        plt.figure()
        plt.imshow(sudoku_a)

    # Preprocessing image to be read
    sudoku_a = cv2.resize(sudoku_a, (450, 450))

    return sudoku_a


# function to greyscale, blur and change the receptive threshold of image

def preprocess(image, display=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 6)
    # blur = cv2.bilateralFilter(gray,9,75,75)
    threshold_img = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    if display:
        plt.figure()
        plt.imshow(threshold_img)
    return threshold_img


def get_outline(sudoku_a, threshold, display=False):
    # Finding the outline of the sudoku puzzle in the image
    contour_1 = sudoku_a.copy()
    contour_2 = sudoku_a.copy()
    contour, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_1, contour, -1, (0, 255, 0), 3)

    if display:
        plt.figure()
        plt.imshow(contour_1)

    return contour, contour_1, contour_2


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


def get_sudoku_grid(sudoku_a, contour, contour_2, display=False):
    black_img = np.zeros((450, 450, 3), np.uint8)
    biggest, maxArea = main_outline(contour)
    if biggest.size != 0:
        biggest = reframe(biggest)
    cv2.drawContours(contour_2, biggest, -1, (0, 255, 0), 10)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imagewrap = cv2.warpPerspective(sudoku_a, matrix, (450, 450))
    imagewrap = cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)

    # cv2.imwrite("data/test2.jpg", imagewrap)
    if display:
        plt.figure()
        plt.imshow(imagewrap)

    return imagewrap


if __name__ == '__main__':
    FOLDER = "data"
    # a = random.choice(os.listdir(folder))
    FILE = "example.jpg"

    SUDOKU_A = read_image(FOLDER, FILE)
    THRESHOLD = preprocess(SUDOKU_A)

    contour, contour_1, contour_2 = get_outline(SUDOKU_A, THRESHOLD)
    biggest, max_area = main_outline(contour)

    imagewrap = get_sudoku_grid(SUDOKU_A, contour, contour_2)
