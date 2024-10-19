import abc
import numpy as np


class LossFunction(abc.ABC):
    @abc.abstractmethod
    def func(self, y, d):
        pass

    @abc.abstractmethod
    def deriv(self, y, d):
        pass


class SquaredError(LossFunction):
    @staticmethod
    def func(y, d):
        return .5 * np.sum((y - d) ** 2)

    @staticmethod
    def deriv(y, d):
        return y - d

