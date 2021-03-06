from matplotlib import pyplot as plt
from utils.display_sudoku import display
from utils.errors import InvalidValue

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

events = dict()


def mouse_event(event):
    return event.xdata, event.ydata


def get_new_point(x, y):
    x = int(x // 100 + 1)
    y = int(y // 100 + 1)
    v = input(f"What is the true value for cell of coordinates (x,y) = ({x},{y}) ? (-1 for empty cell) ")
    try:
        v = int(v)
    except:
        raise InvalidValue(v)
    if v not in [-1] + [i for i in range(1, 10)]:
        raise InvalidValue(v)
    return x, y, v


def on_click_main(event):
    x, y = mouse_event(event)
    try:
        x, y, value = get_new_point(x, y)
        print(f"==> (x : {x}, y : {y}) : new value of {value}")
        index = 9 * (y - 1) + (x - 1)
        events[index] = value
    except InvalidValue as value_error:
        print(value_error.__repr__())


def run_corrections(predictions):
    print("\nClick on a cell to modify its value, or close the window to run the solver.")

    fig = plt.figure("Sudoku")
    plt.axis('off')

    cid = fig.canvas.mpl_connect('button_press_event', on_click_main)

    grid = display(predictions)
    plt.imshow(grid, cmap='Greys')
    plt.show()

    for index in events:
        predictions[index] = events[index]

    if len(list(events.keys())) > 0:
        grid = display(predictions)
        plt.figure("Sudoku")
        plt.axis('off')
        plt.imshow(grid, cmap='Greys')
        plt.show()

    return predictions
