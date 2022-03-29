class CspError(Exception):
    """The generic exception class error to tell that the input CSP has errors."""


class DomainError(CspError):
    """The exception class error to tell that the input CSP has domain errors."""
    def __init__(self, variable):
        """
        Construct the key variable.
        """
        self.variable = variable.name

    def __repr__(self):
        return f"The variable {self.variable} has no domain defined."


class UnknownVariable(CspError):
    """The exception class error to tell that the given variable is unknown."""
    def __init__(self, variable):
        """
        Construct the key variable.
        """
        self.variable = variable.name

    def __repr__(self):
        return f"The given variable {self.variable} is not in the variables list of the CSP."


class ResolutionError(CspError):
    """The exception class error to tell that the input CSP has no solution."""
    def __repr__(self):
        return f"The given Sudoku is not solvable."