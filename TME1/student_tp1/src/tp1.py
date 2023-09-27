
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

        mse = 1/len(y) * torch.sum((yhat - y)**2)
        return mse

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        grad_mse_y = -2/len(y) * (yhat - y) * grad_output
        grad_mse_yhat = 2/len(y) * (yhat - y) * grad_output
        return grad_mse_y, grad_mse_yhat
    

#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE
class Linear(Function):
    """Début d'implementation de la fonction Lineat"""
    @staticmethod
    def forward(ctx, X,W,b):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(X, W, b)
        output = torch.mm(X,W) + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        grad_x = torch.mm(grad_output, ctx.saved_tensors[1].t())
        grad_w = torch.mm(ctx.saved_tensors[0].t(), grad_output)
        grad_b = grad_output.sum(0)
        return grad_x, grad_w,grad_b

## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

