import torch
import matplotlib
from utils import PositionalEncoding
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Initialize the PositionalEncoding module
d_model = 100  
max_len = 100
pos_enc = PositionalEncoding(d_model, max_len)

# Generate positional encodings
pe = pos_enc.pe.squeeze(0)  # Shape: [max_len, d_model]

# Compute the dot product between every pair of positional encodings
similarity = torch.mm(pe, pe.t())  # Shape: [max_len, max_len]

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(similarity.detach().numpy(), cmap='viridis')
plt.title("Positional Encoding Similarity Heatmap")
plt.xlabel("Position")
plt.ylabel("Position")
plt.show()