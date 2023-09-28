import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.05

writer = SummaryWriter()
for n_iter in range(100):
    ##  Calcul du forward (loss)
    loss = MSE.forward(Linear.forward(x, w, b), y)

    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  Calcul du backward (grad_w, grad_b)
    grad_loss_w = Linear.backward(Linear.ctx,(MSE.backward(MSE.ctx,loss)))[1]
    grad_loss_b = Linear.backward(Linear.ctx,(MSE.backward(MSE.ctx,loss)))[2]
    w = w - epsilon * grad_loss_w
    b = b - epsilon * grad_loss_b


