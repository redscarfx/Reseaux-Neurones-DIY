import unittest
from tqdm import tqdm
import numpy as np
import torch
from modules.linear import Linear
from modules.loss import MSELoss

class TestLinear(unittest.TestCase):
    def test_gradient(self, atol = 1e-8, n_iter=25):
        for _ in tqdm(range(n_iter)):
            batch = np.random.randint(100, 300)
            input_size = np.random.randint(5, 30)
            output_size = np.random.randint(1, 15)
            
            ## avec mon module:
            X = np.random.randn(batch, input_size)
            weight = np.random.randn(input_size, output_size) * np.sqrt(2/(input_size+output_size))
            bias = np.random.randn(1, output_size) * np.sqrt(2/(1+output_size))
            
            lin = Linear(input_size, output_size, weight, bias)
            output_custom = lin.forward(X)
            delta = np.ones_like(output_custom)
            lin.zero_grad()
            lin.backward_delta(X, delta)
            lin.backward_update_gradient(X, delta)
            
            ### avec torch:
            X_torch = torch.tensor(X, dtype=torch.float32, requires_grad=False)
            weight_torch = torch.tensor(weight, dtype=torch.float32, requires_grad=True)
            bias_torch = torch.tensor(bias, dtype=torch.float32, requires_grad=True)
            
            output_torch = X_torch.matmul(weight_torch) + bias_torch
            loss = torch.sum(output_torch)
            loss.backward()
            grad_w_torch = weight_torch.grad.detach().numpy()
            grad_b_torch = bias_torch.grad.detach().numpy()
            
            
            self.assertTrue(np.allclose(lin._gradient["w"], grad_w_torch, atol=atol),
                            f"Gradient des poids incorrect pour batch={batch}, input_size={input_size}, output_size={output_size}")
            self.assertTrue(np.allclose(lin._gradient["b"], grad_b_torch, atol=atol),
                            f"Gradient des biais incorrect pour batch={batch}, input_size={input_size}, output_size={output_size}")
            self.assertTrue(np.allclose(output_custom, output_torch.detach().numpy(), atol=atol),
                            f"Sortie incorrecte pour batch={batch}, input_size={input_size}, output_size={output_size}")


if __name__ == '__main__':
    unittest.main()
