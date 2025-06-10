import numpy as np

def sigmoid(z: int | float | np.ndarray) -> float | np.ndarray:
    return 1 / (1 + np.exp(-z))

def d_sigmoid(z: int | float | np.ndarray) -> float | np.ndarray:
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(z: int | float | np.ndarray) -> float | np.ndarray:
    return np.tanh(z)

def d_tanh(z: int | float | np.ndarray) -> float | np.ndarray:
    return 1 - ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))) ** 2

def relu(z: int | float | np.ndarray) -> float | np.ndarray:
    return z * (z > 0)

def d_relu(z):
    return 1. * (z > 0)