from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from utils.generate_dataset import load_dataset, load_kaggle_dataset


def train(X_train, y_train, X_test, y_test, nb_labels=10, output="model3.pth"):
    """
        Defines the model and train it, then stores it as a .pth file.
    """
    # Reshape to be samples*pixels*width*height
    size = X_train.shape[1]
    X_train = X_train.reshape(X_train.shape[0], size, size, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], size, size, 1).astype('float32')

    # One hot Cpde
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Convert from integers to floats
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize to range [0, 1]
    X_train = (X_train / 255.0)
    X_test = (X_test / 255.0)

    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(size, size, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(nb_labels, activation='softmax'))

    # Compile model
    # opt = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])

    model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=5,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_test, y_test),
    )

    # Save model and architecture to single file
    model.save(output)
    print("Saved model to disk")


if __name__ == '__main__':
    DATASETS = ["KAGGLE", "MNIST", "OWN"]
    # DATASET = "MNIST"
    # DATASET = "OWN"
    DATASET = "KAGGLE"

    assert DATASET in DATASETS

    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = None, None, None, None
    OUTPUT = "model3.pth"

    nb_labels = 9
    if DATASET == "MNIST":
        nb_labels = 10
        (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = mnist.load_data()
    elif DATASET == "OWN":
        (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = load_dataset("data/dataset/")
    elif DATASET == "KAGGLE":
        nb_labels = 10
        (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = load_kaggle_dataset("data/kaggle/data/", out_img_size=28, nb_img=2000)

    train(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, nb_labels=nb_labels, output=OUTPUT)
