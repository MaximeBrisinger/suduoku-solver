import os
# To hide tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils.extract_cells import preprocess, get_outline_puzzle, splitcells, cropcell
from utils.post_process import threshold_digit, detect_empty, center_digit
from utils.predict import predict
from utils.modify_digits import run_corrections
from utils.display_sudoku import display
from utils.utils_solver.sudoku import Sudoku
from utils.utils_solver.csp_errors import ResolutionError
from utils.errors import InvalidValue
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import time


def main(file, dataset, model_folder="data/models/", verbose=False):
    """
    Main function. Pre-processes the image, then extracts the digits, recognizes them, and finally solve the sudoku.
    Args:
        file: Name of the .jpg input file of the sudoku.
        dataset: Name of the dataset used for training. Can be "MNIST", "KAGGLE", or "OWN". Best results are obtained
            with KAGGLE dataset.
        model_folder: Path to the folder where model files are stored.
    """

    puzzle = cv2.imread(file)
    su_puzzle = preprocess(puzzle)
    # cv2.imshow('image', su_puzzle)
    # cv2.waitKey(0)

    su_imagewrap = get_outline_puzzle(puzzle, su_puzzle)
    cv2.imshow('Grid', su_imagewrap)
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

    if verbose:
        print(f"Time to solve the sudoku {round(execution_time, 3)} seconds.")
        print(f"Number of branching in the backtracking : {n_branching}")
    final_grid = sudoku.build_solution()
    final_values = np.reshape(final_grid, (1, 81))[0]
    to_display = display(final_values)

    # Show solved Sudoku
    plt.figure("Sudoku")
    plt.axis('off')
    plt.imshow(to_display, cmap='Greys')
    plt.show()


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', default="test4.jpg", type=str)
    parser.add_argument('--dataset', default='KAGGLE', type=str, choices=["KAGGLE", "MNIST", "OWN"],
                        help='Name of the dataset to use for the training.')
    parser.add_argument('--verbose', default=False, help='To print solver indicators at the end.',
                        action="store_true")

    args = parser.parse_args()
    FILE = args.input_file
    DATASET = args.dataset
    VERBOSE = args.verbose

    # Path to data folder
    FOLDER = "data/"

    try:
        main(FOLDER + FILE, DATASET)

    except ResolutionError as res_error:
        print(res_error.__repr__())

    except InvalidValue as value_error:
        print(value_error.__repr__())
