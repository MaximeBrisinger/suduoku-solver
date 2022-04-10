from utils.utils_solver.objects.constraint import Constraint


def alldiff_constraint(variables, domains):
    constraints = list()
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            tuples = list()
            for val_i in domains[variables[i].name]:
                for val_j in domains[variables[j].name]:
                    if val_i != val_j:
                        tuples.append((val_i, val_j))
            constraint = Constraint(variables[i], variables[j], tuples)
            constraints.append(constraint)
    return constraints


def sudoku_constraints(variables, domains):
    constraints = list()
    blocks = sudoku_blocks()

    for i in range(9):
        row = list()
        column = list()
        for j in range(9):
            index = 9 * i + j
            index_sym = 9 * j + i

            row.append(variables[index])
            column.append(variables[index_sym])

        constraints += alldiff_constraint(row, domains)
        constraints += alldiff_constraint(column, domains)

    for block in blocks:
        cells = list()
        for cell in block:
            i, j = cell  # coordinates of the cell
            index = 9 * i + j
            cells.append(variables[index])
        constraints += alldiff_constraint(cells, domains)

    return constraints


def sudoku_blocks():
    blocks = list()
    for i in range(3):
        for j in range(3):
            si, sj = (3 * i, 3 * j)
            block = list()
            for u in range(3):
                for v in range(3):
                    block.append((si + u, sj + v))
            blocks.append(block)
    return blocks




