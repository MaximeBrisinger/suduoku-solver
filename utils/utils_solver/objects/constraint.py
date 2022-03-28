class Constraint:
    def __init__(self, variable1, variable2, tuples):
        self.variables = [variable1, variable2]  # we only consider binary CSP
        self.tuples = tuples

    # def wrapper(self, intensive_constraint):
    # TODO

    def is_satisfied(self, instantiation):
        """
        Check if the constraint is satisfied. If 1 of the 2 variable is not instantiated, return True

        Args
            instantiation (dict[Variable, int]): instantiation (full or partial) for variables of the CSP
        Return
            is_sat (bool) : True if the given instantiation satisfies the constraint

        """
        if self.variables[0].name in instantiation and self.variables[1].name in instantiation:
            v1, v2 = instantiation[self.variables[0].name], instantiation[self.variables[1].name]
            is_sat = (v1, v2) in self.tuples
            return is_sat
        return True




