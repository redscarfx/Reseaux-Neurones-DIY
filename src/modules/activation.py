import numpy as np
from .projet_etu import Module

class TanH(Module):
    def forward(self, X):
        X = np.clip(X, -500, 500)
        return np.tanh(X)

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        tan = self.forward(input)
        return delta * (1-tan**2)
    
    def is_activation(self): 
        return True
    
class Sigmoid(Module):
    def forward(self, X):
        X = np.clip(X, -500, 500)
        return np.divide(1, 1+np.exp(-X))


    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        sig = self.forward(input)
        return delta * sig * (1 - sig)
    
    def is_activation(self): 
        return True  
    
    
class ReLU(Module):
    def forward(self, X):
        zero = np.zeros(X.shape)
        return np.maximum(zero, X)

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return delta * (input > 0)
    
    def is_activation(self): 
        return True




class LeakyReLU(Module):
    r"""Leaky ReLU activation function.

    .. math::
        \text{LeakyReLU}(x) = \max(\alpha x, x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha \times x, & \text{ otherwise }
        \end{cases}
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def zero_grad(self):
        pass

    def forward(self, X):
        return np.maximum(self.alpha * X, X)

    def backward_update_gradient(self, input, delta):
        pass  # No gradient to update in Leaky ReLU

    def backward_delta(self, input, delta):
        r"""
        .. math::
            \frac{\partial M}{\partial z^h} = \begin{cases} 
                1 & \text{if } x>0, \\
                \alpha & \text{otherwise}.
            \end{cases}
        """
        dx = np.ones_like(input)
        dx[input <= 0] = self.alpha
        return delta * dx

    def update_parameters(self, learning_rate):
        pass  # No parameters to update in Leaky ReLU
    
    def is_activation(self):
        return True
