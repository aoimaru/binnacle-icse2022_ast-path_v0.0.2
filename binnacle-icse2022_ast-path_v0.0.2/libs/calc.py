from abc import *

class Calc(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def _do():
        pass