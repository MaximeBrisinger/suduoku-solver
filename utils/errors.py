class InvalidValue(Exception):
    """The exception class error to tell that the value given is invalid."""
    def __init__(self, value):
        """
        Construct the value.
        """
        self.value = value

    def __repr__(self):
        return f"ERROR : The given value {self.value} should be an integer between 1 or 9, or -1 (for empty cell)." \
               f"\nThe modification has not been taken into account."
