import cv2
import numpy as np
import matplotlib.pyplot as plt


def threshold_digit(image, display=False):
    # Input image
    image = cv2.bitwise_not(image, image)

    thresh = 107  # define a threshold, 128 is the middle of black and white in grey scale

    # threshold the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    gray = cv2.bitwise_not(gray, gray)

    if display:
        cv2.imshow("image", gray)
        cv2.waitKey(0)
        # closing all open windows
        cv2.destroyAllWindows()

    return gray


def extreme_white_pixels(image, not_binary=False):
    """
    Among the white pixels of a binary image, finds the minimum and maximum x and y

    Args:
        image(numpy array) : a binary image

    Returns:
        x_min, y_min, x_max, y_max (integers): 2 couples of coordinates, in pixels
    """
    if not_binary:
        image = (image > 50) * 255
    image = (255 - image) / 255
    y_min, x_min = image.shape[0], image.shape[1]

    a = np.linspace(0, x_min - 1, x_min)
    b = np.linspace(0, y_min - 1, y_min)
    c, d = np.meshgrid(a, b)
    ci = c * image
    di = d * image

    return (int(np.min(ci + 2 * x_min * (ci == 0))),
            int(np.min(di + 2 * y_min * (di == 0))),
            int(np.max(ci)),
            int(np.max(di))
            )


def center_digit(gray, not_binary=False, display=False):
    np_img = np.asarray(gray)
    if display:
        plt.imshow(np_img)

    w, h = np.shape(np_img)
    y0, x0, y1, x1 = extreme_white_pixels(np_img, not_binary)
    digit_w = x1 - x0 + 1
    digit_h = y1 - y0 + 1

    x_space = int((x0 + w - x1) / 2)
    y_space = int((y0 + h - y1) / 2)

    new_img = np.zeros((w, h))
    new_img = new_img + 255
    new_img[x_space:x_space + digit_w, y_space:y_space + digit_h] = np_img[x0: x0 + digit_w, y0: y0 + digit_h]

    if display:
        plt.figure()
        plt.imshow(new_img)
        plt.show()

    return new_img


def contours_digit(gray, display=False):
    # Find contours
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        if x < 3 or y < 3 or h < 3 or w < 3:
            # Note the number is always placed in the center
            # Since image is 28x28
            # the number will be in the center thus x >3 and y>3
            # Additionally any of the external lines of the sudoku will not be thicker than 3
            continue
        roi = gray[y:y + h, x:x + w]
        # increasing the size of the number allows better interpretation,
        # try adjusting the number and you will see the difference

    if display:
        cv2.imshow("b", roi)
        cv2.waitKey(0)
        # closing all open windows
        cv2.destroyAllWindows()

    return roi


def detect_empty(gray, threshold=0.06, verbose=False):
    """
    :argument:
        gray (cv2 image) : input image, in black and white mode
    :returns
        Return True if the input image has a digit, False if the cell is empty
    """
    img = np.asarray(gray)
    w, h = np.shape(img)

    xc = int(w * 0.15)
    yc = int(h * 0.15)
    center_img = img[xc: h - xc, yc: w - yc]
    w0, h0 = np.shape(img)

    nb_pixels = h0 * w0
    nb_black = np.sum(center_img < 250)

    ratio_black = nb_black / nb_pixels
    if verbose:
        print(ratio_black)

    return ratio_black > threshold


if __name__ == '__main__':

    files = ['1.jpg', '15.jpg', '16.jpg', '17.jpg', '2.jpg', '20.jpg', '22.jpg', '25.jpg', '26.jpg', '33.jpg', '34.jpg',
             '36.jpg', '37.jpg', '38.jpg', '4.jpg', '40.jpg', '42.jpg', '44.jpg', '45.jpg', '46.jpg', '48.jpg',
             '49.jpg', '56.jpg', '57.jpg', '60.jpg', '62.jpg', '65.jpg', '66.jpg', '67.jpg', '7.jpg', '73.jpg',
             '75.jpg', '78.jpg', '80.jpg', '81.jpg', '9.jpg']

    for i in range(len(files)):
        file = f"data/grid/{files[i]}"
        #file = files[i]
        image = cv2.imread(file)
        im = threshold_digit(image, False)
        # imnp = center_digit(im)

        if detect_empty(im):
            cv2.imwrite(f"data/grid2/{files[i]}", im)