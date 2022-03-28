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


# acj + bci =! j - i ET bcj + aci =! j - i
def queens_columns_constraint(variables, domains, a=1, b=-1):
    constraints = list()
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            tuples = list()
            for val_i in domains[variables[i].name]:
                for val_j in domains[variables[j].name]:
                    if a * val_j + b * val_i != j - i and b * val_j + a * val_i != j - i:
                        tuples.append((val_i, val_j))
            constraint = Constraint(variables[i], variables[j], tuples)
            constraints.append(constraint)
    return constraints


# MARCHE PAS
def queens_binary_var_constraints(variables, nb_columns):
    constraints = list()
    var_treated = list()
    for i1 in range(nb_columns):
        for j1 in range(nb_columns):
            for i2 in range(nb_columns):
                for j2 in range(nb_columns):
                    var_treated.append([(i1, j1), (i2, j2)])
                    var_treated.append([(i2, j2), (i1, j1)])

                    tuples = [(1, 0), (0, 1)]
                    index_1 = i1 * nb_columns + j1
                    index_2 = i2 * nb_columns + j2

                    same_row = (i1 == i2) and (j1 != j2)
                    same_column = (j1 == j2) and (i1 != i2)
                    same_diag = (i1 < i2 and j2 - j1 == i2 - i1) or (i1 < i2 and j1 - j2 == i2 - i1)

                    if same_diag:
                        tuples.append((0, 0))

                    if same_column or same_row or same_diag:
                        #tuples = [(1, 0), (0, 1), (1, 1), (0, 0)]

                        constraint = Constraint(variables[index_1], variables[index_2], tuples)
                        constraints.append(constraint)

    return constraints


# ax + by = c
def weighted_sum_constraint(variables, domains, a, b, c):
    # Useless
    constraints = None
    return constraints


def cartography_constraints(variables, domains, edges):
    constraints = list()
    for edge in edges:
        i, j = edge  # index of the 2 nodes of the edge
        constraints += alldiff_constraint([variables[i - 1], variables[j - 1]],
                                          domains)  # -1 because list first index is 0 (not 1)
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




