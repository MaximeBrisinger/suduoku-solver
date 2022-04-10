from PIL import Image
from utils.crop import *


# Finding the outline of the sudoku puzzle in the image
def get_outline_puzzle(puzzle, su_puzzle, display=False):
    su_contour_1 = su_puzzle.copy()
    su_contour_2 = puzzle.copy()
    su_contour, hierarchy = cv2.findContours(su_puzzle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(su_contour_1, su_contour, -1, (0, 255, 0), 3)
    black_img = np.zeros((450, 450, 3), np.uint8)
    su_biggest, su_maxArea = main_outline(su_contour)
    if su_biggest.size != 0:
        su_biggest = reframe(su_biggest)
    cv2.drawContours(su_contour_2, su_biggest, -1, (0, 255, 0), 10)
    su_pts1 = np.float32(su_biggest)
    su_pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    su_matrix = cv2.getPerspectiveTransform(su_pts1, su_pts2)
    su_imagewrap = cv2.warpPerspective(puzzle, su_matrix, (450, 450))
    su_imagewrap = cv2.cvtColor(su_imagewrap, cv2.COLOR_BGR2GRAY)

    if display:
        plt.figure()
        plt.imshow(su_imagewrap, cmap='Greys')

    return su_imagewrap


def cropcell(cells, display=False):
    cells_croped = []
    c = 0
    for image in cells:
        img = np.array(image)
        # img = img[4:46, 6:46]
        img = img[8:42, 8:42]
        img = Image.fromarray(img)
        cells_croped.append(img)
        c += 1
        if display and c < 10:
            plt.figure()
            plt.imshow(img)
    plt.show()
    return cells_croped


if __name__ == '__main__':
    FOLDER = "data"
    # a = random.choice(os.listdir(folder))
    FILE = "example.jpg"

    SUDOKU_A = read_image(FOLDER, FILE)

    # Importing puzzle to be solved
    puzzle = cv2.imread("../data/test1.jpg")
    su_puzzle = preprocess(puzzle, True)

    su_imagewrap = get_outline_puzzle(puzzle, su_puzzle, SUDOKU_A)

    sudoku_cell = splitcells(su_imagewrap)
    sudoku_cells_croped = cropcell(sudoku_cell, display=True)
    print(np.asarray(sudoku_cells_croped[-1]))

    plt.show()