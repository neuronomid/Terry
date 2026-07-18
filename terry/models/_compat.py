class CallableDict(dict):
    """A property-style Jesse dict that remains callable for Terry 0.1 code."""

    def __call__(self):
        return dict(self)
