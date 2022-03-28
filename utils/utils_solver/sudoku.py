from utils.utils_solver.objects.variable import Variable
from utils.utils_solver.objects.domain import Domain
from utils.utils_solver.objects.wrapper import sudoku_constraints
from utils.utils_solver.csp import CSP
import numpy as np
import matplotlib.pyplot as plt
import time


class Sudoku(CSP):
    def __init__(self, grid):

        # Variables
        variables = [Variable("x" + str(i) + "," + str(j))
                     for i in range(1, 10)
                     for j in range(1, 10)]

        # Domains
        domains = Domain(variables)
        domains.fill_all_domains_by_range(lb=1, ub=9)

        # Constraints
        constraints = sudoku_constraints(variables, domains.dict)

        self.pre_assigned = dict()
        for i in range(9):
            for j in range(9):
                value = grid[i, j]
                if value != -1:
                    self.pre_assigned[(i + 1, j + 1)] = value
                    domains.dict["x" + str(i + 1) + "," + str(j + 1)] = [value]

        super().__init__(variables=variables, domains=domains, constraints=constraints)

    def build_solution(self):
        n = 9
        grid = np.zeros((n, n))
        for var in self.final_solution.keys():
            coordo = var.split("x")[1].split(",")
            i, j = int(coordo[0]) - 1, int(coordo[1]) - 1
            grid[i][j] = self.final_solution[var]

        return grid

        # plt.figure(f"Sudoku solution :")
        # plt.imshow(grid, cmap='Pastel1')
        # # fig.axes.get_xaxis().set_visible(False)
        # # fig.axes.get_yaxis().set_visible(False)
        #
        # ax = plt.gca()
        #
        # # Major ticks
        # ax.set_xticks([])
        # ax.set_yticks([])
        #
        # # Minor ticks
        # ax.set_xticks(np.arange(-.5, n, 3), minor=True)
        # ax.set_yticks(np.arange(-.5, n, 3), minor=True)
        #
        # # Gridlines based on minor ticks
        # ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        #
        # plt.colorbar()
        # plt.show()

    def show_pre_assigned(self):
        n = 9
        grid = np.empty((n, n))
        grid[:] = np.NaN
        for var in self.pre_assigned.keys():
            i, j = var
            grid[i - 1][j - 1] = self.pre_assigned[var]

        plt.figure(f"Sudoku pre assigned values")
        plt.imshow(grid, cmap='Pastel1')
        # fig.axes.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)

        ax = plt.gca()

        # Major ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Minor ticks
        ax.set_xticks(np.arange(-.5, n, 3), minor=True)
        ax.set_yticks(np.arange(-.5, n, 3), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

        plt.colorbar()
        plt.show()

