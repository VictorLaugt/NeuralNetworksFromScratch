import abc
import numpy as np


class ActivationFunction(abc.ABC):
    @abc.abstractmethod
    def func(self, a):
        pass

    @abc.abstractmethod
    def deriv(self, a):
        pass

    # @abc.abstractmethod
    # def init_parameters(self, n_input):
    #     ...


class Sigmoid(ActivationFunction):
    @staticmethod
    def func(a):
        return 1 / (1 + np.exp(-a))

    @staticmethod
    def deriv(a):
        sigmoid_a = 1 / (1 + np.exp(-a))
        return sigmoid_a * (1 - sigmoid_a)


class ReLu(ActivationFunction):
    @staticmethod
    def func(a):
        return np.max(a, 0.)

    @staticmethod
    def deriv(a):
        result = np.ones_like(a)
        result[a < 0.] = 0.
        return result


class LeakyRelu(ActivationFunction):
    def __init__(self, slope):
        self.slope = slope

    def func(self, a):
        result = np.empty_like(a)
        positive_mask = (a > 0)
        negative_mask = ~positive_mask
        result[positive_mask] = a[positive_mask]
        result[negative_mask] = self.slope * a[negative_mask]
        return result

    def deriv(self, a):
        result = np.ones_like(a)
        result[a < 0.] = self.slope
        return result
