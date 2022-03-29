import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils.extract_cells import preprocess, get_outline_puzzle, splitcells, cropcell
from utils.post_process import threshold_digit, detect_empty, center_digit
from utils.predict import predict
from utils.modify_digits import run_corrections
from utils.display_sudoku import display
from utils.utils_solver.sudoku import Sudoku
from utils.utils_solver.csp_errors import ResolutionError
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import time


def main(file, dataset, model_folder="data/models/"):

    puzzle = cv2.imread(file)
    su_puzzle = preprocess(puzzle)
    cv2.imshow('image', su_puzzle)
    cv2.waitKey(0)

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
            elif dataset == "MNIST":
                model = model_folder + "model.pth"
                pred = predict(image, model=model, img_size=28)
                pred_value = pred[0, 1:].argmax() + 1
            else:
                image = 255 - image
                model = model_folder + "model3.pth"
                pred = predict(image, model=model, img_size=34)
                pred_value = pred[0].argmax()

            predicted.append(pred_value)

    predicted = run_corrections(predicted)
    predicted = np.reshape(predicted, (9, 9))

    # Solve Sudoku grid
    sudoku = Sudoku(predicted)

    print(f"\nSolving Sudoku...")
    solution, termination_status, execution_time, n_branching = sudoku.main(instantiation=dict(),
                                                                            start=time.time(),
                                                                            mode_var_heuristic=1,
                                                                            mode_val_heuristic=1,
                                                                            arc_consistence=True,
                                                                            forward_check=True,
                                                                            time_limit=180)
    try:
        final_grid = sudoku.build_solution()
        final_values = np.reshape(final_grid, (1, 81))[0]
        to_display = display(final_values)

        # Show solved Sudoku
        plt.figure("Sudoku")
        plt.axis('off')
        plt.imshow(to_display, cmap='Greys')
        plt.show()

    except ResolutionError as error:
        print(error.__repr__())


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

