from utils.extract_cells import *
from old.digit_classification import *
from utils.post_process import *
import numpy as np
# import pytesseract


FOLDER = "data"
# a = random.choice(os.listdir(folder))
FILE = "example.jpg"
#
# img = Image.open("digits/0.jpg")
# img = np.array(255 - np.asarray(img))
# img = Image.fromarray(img)
# img.save("digits/00.jpg")
# exit()

SUDOKU_A = read_image(FOLDER, FILE)

# Not sure if mandatory
puzzle = cv2.imread("../data/test1.jpg")
su_puzzle = preprocess(puzzle)


su_imagewrap = get_outline_puzzle(puzzle, su_puzzle, SUDOKU_A)
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
exit()

#################################################
X_train, y_train, clf = train()
x_test, predicted, scores = test(sudoku_cells_croped, clf)


for i in range(len(predicted)):
    if max(scores[i]) < 0.20:
        predicted[i] = -1

check_predictions(x_test, predicted)

predicted = np.reshape(predicted, (9, 9))
print(predicted)
print(scores[0])

# options = "outputbase digits"
# text = pytesseract.image_to_string("digits/1.jpg", config=options)
# print(f"pytess : {text}")