from .projet_etu import *
import numpy as np

class Linear(Module):
    def __init__(self, input, output, weight=None, bias=None):
        super().__init__()
        self._parameters = {}
        self._gradient = {}
        self._output = output
        self._input = input
        if weight is not None:
            self._parameters["w"] = weight
        else:
            self._parameters["w"] = np.random.randn(input, output) * np.sqrt(2/(input+output))
        if bias is not None:
            self._parameters["b"] = bias
        else:
            self._parameters["b"] = np.random.randn(1, output) * np.sqrt(2/(1+output))
        self._gradient["w"] = np.zeros_like(self._parameters["w"])
        self._gradient["b"] = np.zeros_like(self._parameters["b"])
 
         
    def forward(self, X):
        return X @ self._parameters["w"] + self._parameters["b"]
    
    def backward_update_gradient(self, input, delta):
        self._gradient["w"] +=  input.T @ delta
        self._gradient["b"] += np.sum(delta, axis=0)

    def backward_delta(self, input, delta):
        return delta @ self._parameters["w"].T

    
    def update_parameters(self, gradient_step=0.001):
        self._parameters["w"] -= gradient_step * self._gradient["w"]
        self._parameters["b"] -= gradient_step * self._gradient["b"]
        
    def zero_grad(self):
        self._gradient["w"] = np.zeros_like(self._parameters["w"])
        self._gradient["b"] = np.zeros_like(self._parameters["b"])

    def is_activation(self):
        return False 