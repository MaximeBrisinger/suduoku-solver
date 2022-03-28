from utils.extract_cells import *
from utils.post_process import *
import numpy as np


FOLDER = "data/"
FILE = "test1.jpg"

puzzle = cv2.imread(FOLDER + FILE)
su_puzzle = preprocess(puzzle)


su_imagewrap = get_outline_puzzle(puzzle, su_puzzle)
sudoku_cell = splitcells(su_imagewrap)
sudoku_cells_croped = CropCell(sudoku_cell, display=True)


predicted = []
c = 0
for cell in sudoku_cells_croped:
    c += 1
    cell = cv2.cvtColor(np.array(cell), cv2.COLOR_RGB2BGR)
    img = threshold_digit(cell)
    has_digit = detect_empty(img)
    value = 1 if has_digit else 0
    if has_digit:
        cv2.imwrite(f"data/grid2/{c}.jpg", img)
    predicted.append(value)

predicted = np.reshape(predicted, (9, 9))
print(predicted)
