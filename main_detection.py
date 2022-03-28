from utils.extract_cells import preprocess, get_outline_puzzle, splitcells, cropcell
from utils.post_process import threshold_digit, detect_empty, center_digit
from utils.predict import predict
from utils.modify_digits import run_corrections
import numpy as np
import cv2
import argparse
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(file, dataset, model_folder="data/models/"):

    puzzle = cv2.imread(file)
    su_puzzle = preprocess(puzzle)
    # cv2.imshow('image', su_puzzle)
    # cv2.waitKey(0)

    su_imagewrap = get_outline_puzzle(puzzle, su_puzzle)
    cv2.imshow('image', su_imagewrap)
    cv2.waitKey(0)

    sudoku_cell = splitcells(su_imagewrap)
    sudoku_cells_croped = cropcell(sudoku_cell, display=False)

    predicted = []
    c = 0
    for cell in sudoku_cells_croped:

        c += 1
        cell = cv2.cvtColor(np.array(cell), cv2.COLOR_RGB2BGR)
        img = threshold_digit(cell)

        # Check if empty
        has_digit = detect_empty(img)

        if not has_digit:
            predicted.append(-1)
        else:
            image = center_digit(img)

            # To save extracted cells as jpg (to analyze performances)
            # img = Image.fromarray(image).convert('RGB')
            # img.save(f"data/test/digits_test2/{c}.jpg")

            if dataset == "OWN":
                model = model_folder + "model2.pth"
                pred = predict(image, model=model, img_size=28)
                pred_value = pred[0].argmax() + 1
                # print(pred_value)
            elif dataset == "MNIST":
                model = model_folder + "model.pth"
                pred = predict(image, model=model, img_size=28)
                pred_value = pred[0, 1:].argmax() + 1
                # print(pred_value)
            else:
                image = 255 - image
                model = model_folder + "model3.pth"
                pred = predict(image, model=model, img_size=34)
                pred_value = pred[0].argmax()
                # print(pred_value)

            predicted.append(pred_value)

    # print(predicted)
    # print(f"Nb of 8 : {sum(np.array(predicted) == 8)} / {len(predicted)}")
    # print(np.reshape(predicted, (9, 9)))

    run_corrections(predicted)


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', default="test4.jpg", type=str)
    parser.add_argument('--dataset', default='KAGGLE', type=str, choices=["KAGGLE", "MNIST", "OWN"],
                        help='Name of the dataset to use for the training.')

    args = parser.parse_args()
    FILE = args.input_file
    DATASET = args.dataset

    FOLDER = "data/"

    main(FOLDER + FILE, DATASET)

