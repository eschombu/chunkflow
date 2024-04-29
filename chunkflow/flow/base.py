from abc import ABCMeta, abstractmethod


class OperatorBase(metaclass=ABCMeta):
    """Real Operator should inherit from this base class."""
    def __init__(self, name: str = None):
        assert isinstance(name, str)
        self.name = name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """The processing should happen in this function."""
        pass
