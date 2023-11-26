import torch

# Supposons que 'probabilities' est votre tensor de taille [100, 10] avec des probabilités de classe
# Remplacer cela par votre tensor réel de probabilités
probabilities = torch.rand(100, 10)

# Calculer l'entropie pour chaque vecteur de probabilité dans le batch
entropies = -(probabilities * probabilities.log()).mean(dim=0)
p = torch.special.entr(probabilities).mean(dim=0)
# 'entropies' est maintenant un tensor de taille [100], avec l'entropie de chaque vecteur de probabilités
print(entropies)
print(p.shape)
print(p)