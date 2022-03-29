from utils.utils_solver.objects.variable import Variable
from utils.utils_solver.objects.domain import Domain
from utils.utils_solver.objects.wrapper import sudoku_constraints
from utils.utils_solver.csp import CSP
from utils.utils_solver.csp_errors import ResolutionError
import numpy as np
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
        try:
            for var in self.final_solution.keys():
                coordo = var.split("x")[1].split(",")
                i, j = int(coordo[0]) - 1, int(coordo[1]) - 1
                grid[i][j] = self.final_solution[var]

            return grid
        except:
            raise ResolutionError()

