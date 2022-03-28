from utils.post_process import center_digit, threshold_digit
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.optimizers import Adam
import cv2
import numpy as np
import matplotlib.pyplot as plt


def train():
    # the MNIST data is split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape to be samples*pixels*width*height
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

    # One hot Cpde
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # convert from integers to floats
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # normalize to range [0, 1]
    X_train = (X_train / 255.0)
    X_test = (X_test / 255.0)


    model = Sequential()
    if True:
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
    else:
        model.add(Conv2D(32, (3, 3), activation='tanh', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='tanh', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='tanh', kernel_initializer='he_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
    # model.summary()

    # compile model
    opt = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=10,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_test, y_test),
    )

    # save model and architecture to single file
    model.save("model1.pth")
    print("Saved model to disk")


def predict(img):
    image = img.copy()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)[1]
    image = cv2.resize(image, (28, 28))
    # display_image(image)
    image = image.astype('float32')

    image = image.reshape(1, 28, 28, 1)
    image /= 255.0

    plt.imshow(image.reshape(28, 28), cmap='Greys')
    plt.show()
    model = load_model("../data/models/model.pth")
    pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)
    pred_value = pred[0, 1:].argmax() + 1
    print(pred_value)
    return pred


#train()

files = ['1.jpg', '15.jpg', '16.jpg', '17.jpg', '2.jpg', '20.jpg', '22.jpg', '25.jpg', '26.jpg', '33.jpg', '34.jpg', '36.jpg', '37.jpg', '38.jpg', '4.jpg', '40.jpg', '42.jpg', '44.jpg', '45.jpg', '46.jpg', '48.jpg', '49.jpg', '56.jpg', '57.jpg', '60.jpg', '62.jpg', '65.jpg', '66.jpg', '67.jpg', '7.jpg', '73.jpg', '75.jpg', '78.jpg', '80.jpg', '81.jpg', '9.jpg']


def predict_list_files(list_files):
    predicted = []
    folder = "data/grid/"

    for file_name in list_files:
        file = folder + file_name

        image = cv2.imread(file)

        image = threshold_digit(image)
        image = center_digit(image)
        pred = predict(image)

        pred_value = pred[0, 1:].argmax() + 1
        # print(pred_value)

        predicted.append(pred_value)

    print(predicted)
    print(f"Nb of 8 : {sum(np.array(predicted)==8)} / {len(predicted)}")
    return predicted


if __name__ == '__main__':
    predict_list_files(files)