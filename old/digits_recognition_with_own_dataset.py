from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from utils.generate_dataset import load_dataset


def train():
    (X_train, y_train), (X_test, y_test) = load_dataset("data/dataset/")

    # Reshape to be samples*pixels*width*height
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

    # One hot Cpde
    y_train = np_utils.to_categorical(y_train - 1)
    y_test = np_utils.to_categorical(y_test - 1)
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
        model.add(Dense(9, activation='softmax'))

    # compile model
    # opt = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])

    model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=10,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_test, y_test),
    )

    # save model and architecture to single file
    model.save("model2.pth")
    print("Saved model to disk")

