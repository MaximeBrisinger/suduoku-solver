import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


def train():
    digits = datasets.load_digits()

    # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    # for ax, image, label in zip(axes, digits.images, digits.target):
    #     ax.set_axis_off()
        # ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        # ax.set_title("Training: %i" % label)

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001, probability=True)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.1, shuffle=False
    )

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    return X_train, y_train, clf


def test(images, clf):
    x_test = []
    for image in images:
        image = image.resize((8, 8))
        # img = np.array((np.asarray(image)) / 255 * 16, dtype=np.int16).flatten()
        img = np.array((255 - np.asarray(image)) / 255 * 16, dtype=np.int16).flatten()

        x_test.append(img)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(x_test)
    scores = clf.predict_proba(x_test)

    return x_test, predicted, scores


def check_predictions(x_test, predicted):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    plt.show()


if __name__ == '__main__':
    img = Image.open("data/2.jpg")
    X_train, y_train, clf = train()
    x_test, predicted, scores = test([img], clf)
    check_predictions(x_test, predicted)
