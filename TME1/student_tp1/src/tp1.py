import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """

    def __init__(self):
        self._saved_tensors = ()

    def save_for_backward(self, *args):
        self._saved_tensors = args

    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""

    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)
        return (yhat - y) ** 2 

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        d_yhat = grad_output * 2 * (yhat - y)
        d_y = grad_output * -2 * (yhat - y)
        return d_yhat, d_y


class Linear(Function):
    @staticmethod
    def forward(ctx, X, W, b):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(X, W, b)
        return X @ W + b

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        X, W, _ = ctx.saved_tensors
        grad_x = grad_output @ W.T  # (n, d)
        grad_w = X.T @ grad_output  # (d, p)
        grad_b = grad_output  # (n, p)
        return grad_x, grad_w, grad_b


## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply