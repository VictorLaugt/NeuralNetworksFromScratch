import abc
import numpy as np


class ActivationFunction(abc.ABC):
    @abc.abstractmethod
    def func(self, a):
        pass

    @abc.abstractmethod
    def deriv(self, a):
        pass

    @staticmethod
    def layer_init(rng, n_in, n_out):
        W = rng.standard_normal((n_out, n_in))
        b = rng.standard_normal((n_out, 1))
        return W, b


class Sigmoid(ActivationFunction):
    @staticmethod
    def func(a):
        return 1 / (1 + np.exp(-a))

    @staticmethod
    def deriv(a):
        sigmoid_a = 1 / (1 + np.exp(-a))
        return sigmoid_a * (1 - sigmoid_a)

    @staticmethod
    def layer_init(rng, n_in, n_out):
        """Xavier (Glorot) weights initialization"""
        W = rng.normal(loc=0., scale=np.sqrt(1 / n_in), size=(n_out, n_in))
        b = np.zeros((n_out, 1))
        return W, b


class ReLu(ActivationFunction):
    @staticmethod
    def func(a):
        return np.max(a, 0.)

    @staticmethod
    def deriv(a):
        result = np.ones_like(a)
        result[a < 0.] = 0.
        return result

    @staticmethod
    def layer_init(rng, n_in, n_out):
        """He weights initialization"""
        W = rng.normal(loc=0., scale=np.sqrt(2 / n_in), size=(n_out, n_in))
        b = np.zeros((n_out, 1))
        return W, b


class LeakyRelu(ReLu):
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
