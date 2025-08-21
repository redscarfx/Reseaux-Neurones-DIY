from .projet_etu import Loss
import numpy as np

class MSELoss(Loss):        
    def forward(self, y, yhat):
        return ((y-yhat)**2)
    
    def backward(self, y, yhat):
        return  -2* (y-yhat)
    

class CELossLogSoftmax:
    def forward(self, y, y_hat):
        #y en one-hot, y_hat valeurs en R^k (k=nb_classes) avant softmax
        y_hat -= np.max(y_hat, axis=1, keepdims=True)
        log_softmax = y_hat - np.log(np.exp(y_hat).sum(axis=1, keepdims=True))
        loss = -np.sum(y * log_softmax) / y.shape[0]
        return loss
    
    def backward(self, y, y_hat):
        # ∂L/∂z  =softmax(z)−y avec y en one hot
        y_hat -= np.max(y_hat, axis=1, keepdims=True)
        softmax = np.exp(y_hat) / np.exp(y_hat).sum(axis=1, keepdims=True)
        grad = (softmax - y) / y.shape[0]
        return grad
    

class BCELoss:
    def forward(self, y, y_hat):
        epsilon = 1e-5
        #y en one-hot, y_hat valeurs en [0,1] (apres sigmoid
        loss = -np.mean(y *np.log(y_hat+epsilon) + (1 - y) *  np.log(1 - y_hat+epsilon))
        return loss
    
    def backward(self, y, y_hat):
        # ∂L/∂z  =y_hat−y avec y en one hot
        grad = (y_hat - y) / y.shape[0]
        return grad
    