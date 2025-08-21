import numpy as np
import unittest
import torch
from tqdm import tqdm

from modules.activation import TanH, Sigmoid


class TestActivation(unittest.TestCase):
    def test_gradient_tanh(self, atol=1e-5, n_iter=25):
        for _ in tqdm(range(n_iter)):
            np.random.seed(42) 
            X = np.random.randn(10, 10)
            activation = TanH()
            delta = np.ones_like(X)
            grad_analytic = activation.backward_delta(X, delta)  
            epsilon = 1e-10
            grad_numeric = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    x = X[i, j]
                    f_plus = np.tanh(x + epsilon)
                    f_minus = np.tanh(x - epsilon)
                    grad_numeric[i, j] = (f_plus - f_minus) / (2 * epsilon)
            self.assertTrue(np.allclose(grad_analytic, grad_numeric, atol=atol),
                            f"Le gradient de TanH ne correspond pas à l'approximation numérique (tolérance={atol}).")
        
    def test_gradient_sigmoid(self, atol=1e-5, n_iter=25):
        for _ in tqdm(range(n_iter)):
            np.random.seed(42)  
            X = np.random.randn(10, 10)
            activation = Sigmoid()
            delta = np.ones_like(X)
            grad_analytic = activation.backward_delta(X, delta)
            epsilon = 1e-10
            grad_numeric = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    x = X[i, j]
                    x_plus = np.clip(x + epsilon, -500, 500)
                    x_minus = np.clip(x - epsilon, -500, 500)
                    f_plus = 1 / (1 + np.exp(-x_plus))
                    f_minus = 1 / (1 + np.exp(-x_minus))
                    grad_numeric[i, j] = (f_plus - f_minus) / (2 * epsilon)
            self.assertTrue(np.allclose(grad_analytic, grad_numeric, atol=atol),
                            f"Le gradient de Sigmoid ne correspond pas à l'approximation numérique (tolérance={atol}).")

if __name__ == '__main__':
    unittest.main()
