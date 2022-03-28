class Domain:
    def __init__(self, variables):
        self.variables = variables
        self.dict = {variable.name: [] for variable in variables}

    def fill_domain_with_values(self, variable, values):
        """
        Build a domain for given variable with explicit values.

        Args:
            variable (Variable)
            values (list of int) : possible values for variable
        """
        self.dict[variable.name] = values

    def fill_domain_by_range(self, variable, lb, ub):
        """
        Build a linear domain for given variable, from lower bound lb to upper bound ub.

        Args:
            variable (Variable)
            lb (int) : lower bound
            ub (int) : upper bound
        """
        self.dict[variable.name] = [x for x in range(lb, ub + 1)]

    def fill_all_domains_by_range(self, lb, ub):
        """
        Build a linear domain for all variables, from lower bound lb to upper bound ub.

        Args:
            lb (int) : lower bound
            ub (int) : upper bound
        """
        for variable in self.variables:
            self.dict[variable.name] = [x for x in range(lb, ub + 1)]
