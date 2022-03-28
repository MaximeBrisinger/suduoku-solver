from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import cv2
import random
import numpy as np
from utils.post_process import center_digit
import matplotlib.pyplot as plt


# plot digit numbers (from 1 to 9):
def draw_digits(x_pos, y_pos, size, font_path, out_folder):
    for i in range(1, 10):
        img = Image.new('L', (28, 28))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, size)
        draw.text((x_pos, y_pos), str(i), 255, font=font)
        img = np.array(255 - np.asarray(img))
        img = center_digit(img, not_binary=True)
        img = 255 - img
        cv2.imwrite(out_folder + str(i) + "_" + str(font_path.split("/")[-1].split(".ttf")[0]) + str(x_pos) + str(-y_pos) + str(size) + '.jpg', img)


# TODO : add noise to dataset and sharpen
def load_dataset(out_folder, test_prop=0.2):
    data = os.listdir(out_folder)
    random.shuffle(data)

    x_train, y_train, x_test, y_test = list(), list(), list(), list()

    for img_file in data:
        img = cv2.imread(out_folder + '/' + img_file, cv2.IMREAD_GRAYSCALE)

        label = int(img_file.split("_")[0])
        if len(x_train) / len(data) < 1 - test_prop:
            x_train.append(img)
            y_train.append(label)
        else:
            x_test.append(img)
            y_test.append(label)

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


def preprocess_kaggle_images(img, out_img_size):
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
    h, w, *_ = img.shape
    h_padding = max((out_img_size - h) // 2, 0)
    w_padding = max((out_img_size - w) // 2, 0)
    img = cv2.copyMakeBorder(img, h_padding, h_padding, w_padding, w_padding, cv2.BORDER_CONSTANT)
    img = cv2.resize(img, (out_img_size, out_img_size))
    img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT)
    return img


def load_kaggle_dataset(out_folder, out_img_size=28, display=False, nb_img=200):

    train_folder = out_folder + "training_data/"

    test_folder = out_folder + "testing_data/"
    test_files = os.listdir(test_folder)

    digits = [str(i) for i in range(1, 10)]

    random.shuffle(test_files)

    x_train, y_train, x_test, y_test = list(), list(), list(), list()

    for digit in digits:
        c = 0
        folder = train_folder + digit
        files = os.listdir(folder)
        random.shuffle(files)
        for img_file in files[:nb_img]:
            img = cv2.imread(folder + '/' + img_file, cv2.IMREAD_GRAYSCALE)
            img = preprocess_kaggle_images(img, out_img_size)

            c += 1
            save_img_folder ="data/test/digits_kaggle_postprocessed/"
            if c < 5:
                cv2.imwrite(f"{save_img_folder}{digit}{c}.jpg", img)

            if display:
                cv2.imshow('image', img)
                cv2.waitKey(0)

            label = int(digit)
            x_train.append(img)
            y_train.append(label)

    for digit in digits:
        folder = test_folder + digit
        files = os.listdir(folder)
        random.shuffle(files)
        for img_file in files[:nb_img//5]:
            img = cv2.imread(folder + '/' + img_file, cv2.IMREAD_GRAYSCALE)
            img = preprocess_kaggle_images(img, out_img_size)

            label = int(digit)
            x_test.append(img)
            y_test.append(label)

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


if __name__ == '__main__':
    out_folder = "data/dataset/"
    out_folder_kaggle = "../data/kaggle/data/"

    load_kaggle_dataset(out_folder_kaggle)
    exit()

    # Fonts
    fonts_folder = "data/fonts/"
    fonts = os.listdir(fonts_folder)

    # x and y
    # x_list = [6, 7, 8]
    # y_list = [-2, -3, -4]
    sizes = [20, 22, 24, 26]

    for font in fonts:
        font = fonts_folder + font
        for x in [7]:
            for y in [-3]:
                for size in sizes:
                    draw_digits(x, y, size, font, out_folder)