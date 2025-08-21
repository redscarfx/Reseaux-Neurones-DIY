import numpy as np
import sys
from .projet_etu import *
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from .loss import CELossLogSoftmax
import pickle
import pandas as pd


class Sequential(Module):
    def __init__(self, *sequence):
        if len(sequence) == 0:
            raise Exception("Empty inialization")
        if len(sequence) == 1 and type(sequence[0]) == list:
            self._layers = copy.deepcopy(list(sequence[0]))
        else:
            self._layers = copy.deepcopy(list(sequence))
        self._forwards = []
        self._deltas = []
        
    
    def append(self, module):
        self._layers.append(module)
        
    def print_layers(self):
        print([layer.__class__.__name__ for layer in self._layers])           
            
    def zero_grad(self):
        for layer in self._layers:
            layer.zero_grad()

    
    def add(self, module, i=0):
        self._layers.append(module)
    
    def pop(self, i=-1):
        return self._layers.pop(i)
    
    def forward(self, X):
        self._inputs = [X] 
        out = X
        for module in self._layers:
            self._inputs.append(out) 
            out = module.forward(out)
        return out

    
    def get_output(self):
        return self._forwards[-1][1]

    def update_parameters(self, gradient_step=1e-3):
        for layer in self._layers:
            layer.update_parameters(gradient_step)

    def backward(self, X, delta):
        for module, input_layer in zip(reversed(self._layers), reversed(self._inputs)):
            if not module.is_activation():
                module.backward_update_gradient(input_layer, delta)
            delta = module.backward_delta(input_layer, delta)
        return delta


class Optim(object):
    def __init__(self, model_sequenciel, loss, eps):
        self.model_sequentiel = model_sequenciel
        self.eps = eps
        self.loss = loss
        self.last_forward = None

    def step(self, batch_x, batch_y, batch_x_test, batch_y_test,eps=None,):
        if eps is not None:
            self.eps=eps
        forward = self.model_sequentiel.forward(batch_x)
        self.last_forward = forward
        loss_value = self.loss.forward(batch_y, forward)
        loss_delta = self.loss.backward(batch_y, forward)
        self.model_sequentiel.zero_grad()
        self.model_sequentiel.backward(batch_x, loss_delta)
        self.model_sequentiel.update_parameters(self.eps)
        if batch_x_test is not None and batch_y_test is not None:
            forward_test = self.model_sequentiel.forward(batch_x_test)
            loss_value_test = self.loss.forward(batch_y_test, forward_test)
        if batch_x_test is not None and batch_y_test is not None:
            return loss_value, loss_value_test
        return loss_value, None
    
    def test(self, X_test, y_test):
        pred = self.model_sequentiel.forward(X_test)
        loss = self.loss.forward(y_test, pred)
        return loss, pred
    
    def SGD(self, X_train, y_train, batch_size, n_iterations, gradient_step= None):
        if gradient_step is None:
            gradient_step = self.eps
        loss_list = []
        test_loss_list = []
        pred_list = []
        for _ in range(n_iterations):
            loss_list.append(0)
            pred_list.append(0)
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_shuffled[i:i+batch_size]
                batch_y = y_train_shuffled[i:i+batch_size]
                loss = self.step(batch_X,batch_y, gradient_step)
                loss_list[-1] += loss
                pred_list[-1] += self.last_forward
        return loss_list, pred_list

    def print_layers(self):
        self.model_sequentiel.print_layers()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

def multi_class(
    X_train, y_train, X_test, y_test, 
    model, 
    loss=CELossLogSoftmax(), 
    n_iter=1000, 
    eps=1e-2, 
    classes_names=None, 
    titre="Classification multi-classe", model_name="models/model", loss_name="loss.csv",
):
    # y_train et y_test sont supposés être encodés en one-hot
    batch, input_dim, hidden1, hidden2, n_classes = X_train.shape[0], X_train.shape[1], 10, 4, y_train.shape[1]
    loss_list = []
    test_loss_list = []

    optim = Optim(model, loss, eps)
    
    for _ in tqdm(range(n_iter)):
        x,y = optim.step(X_train, y_train, X_test, y_test)
        loss_list.append(x)
        test_loss_list.append(y)
        

    if classes_names is None:
        classes_names = [str(i) for i in range(n_classes)]

    # Prédictions sur le test
    y_pred = optim.test(X_test, y_test)[1]
    y_true = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred_labels)
    
    # Création de la figure avec deux sous-graphes
    fig, axes = plt.subplots(1, 2, figsize=(8,4 ))
    
    # Sous-graphe 1 : courbes de loss
    sns.lineplot(ax=axes[0], x=range(n_iter), y=loss_list, label='Train Loss', color='blue')
    sns.lineplot(ax=axes[0], x=range(n_iter), y=test_loss_list, label='Test Loss', color='orange')
    axes[0].set_title("Courbes de perte", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Itérations")
    axes[0].set_ylabel("Valeur de la perte")
    axes[0].legend()
    axes[0].grid(True)

    # Sous-graphe 2 : matrice de confusion
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], 
                xticklabels=classes_names, yticklabels=classes_names, cbar=False)
    axes[1].set_title("Matrice de confusion", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Prédit")
    axes[1].set_ylabel("Vrai")

    # Titre global
    fig.suptitle(titre, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Rapport de classification
    print("\nRapport de classification :\n")
    print(classification_report(y_true, y_pred_labels, target_names=classes_names))
    
    #save model
    pickle.dump(model, open(model_name, 'wb'))
    
    #save loss of train and test as csv
    loss_df = pd.DataFrame({'Train Loss': loss_list, 'Test Loss': test_loss_list})
    loss_df.to_csv(loss_name, index=False)
    
