from utils.post_process import center_digit, threshold_digit
from keras.models import load_model
import cv2
import numpy as np
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def predict(img, model, img_size):
    image = img.copy()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)[1]
    image = cv2.resize(image, (img_size, img_size))
    # display_image(image)
    image = image.astype('float32')

    image = image.reshape(1, img_size, img_size, 1)
    image /= 255.0

    # plt.imshow(image.reshape(28, 28), cmap='Greys')
    # plt.show()
    model = load_model(model)
    pred = model.predict(image.reshape(1, img_size, img_size, 1), batch_size=1)
    return pred


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