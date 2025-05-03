import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder: turns inputs into embeddings (fingerprints)
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Contrastive loss: pulls positives together, pushes negatives apart
def contrastive_loss(emb1, emb2, temperature):
    # Normalize embeddings to unit length
    emb1_norm = F.normalize(emb1, dim=1)
    emb2_norm = F.normalize(emb2, dim=1)
    # Cosine similarity between every pair
    sim_matrix = torch.matmul(emb1_norm, emb2_norm.T) / temperature

    # For each anchor i, the correct positive is at index i
    batch_size = emb1.size(0)
    labels = torch.arange(batch_size, device=emb1.device)

    # Softmax + cross entropy:
    #   softmax turns raw similarities into probabilities
    #   cross entropy penalizes if the positive is not the top match
    loss_i = F.cross_entropy(sim_matrix, labels)
    loss_j = F.cross_entropy(sim_matrix.T, labels)
    return 0.5 * (loss_i + loss_j)

def main():
    torch.manual_seed(0)

    # 1) Create a toy batch of inputs
    batch_size, input_dim = 4, 5
    x = torch.randn(batch_size, input_dim)

    # 2) Make two "views" (anchor and positive) by adding small noise
    view_anchor   = x + 0.1 * torch.randn_like(x)  # anchor
    view_positive = x + 0.1 * torch.randn_like(x)  # positive

    # 3) Set up encoder and optimizer
    encoder   = Encoder(input_dim, hidden_dim=10, embed_dim=3)  # encoder generates embeddings
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

    # 4) Forward pass: get embeddings
    emb_anchor   = encoder(view_anchor)    # embeddings for anchor
    emb_positive = encoder(view_positive)  # embeddings for positive

    # 5) Define temperature knob
    temperature = 0.07

    # 6) Compute the contrastive loss
    loss = contrastive_loss(emb_anchor, emb_positive, temperature)
    print("Contrastive Loss:", loss.item())

    # 7) Backpropagation: compute gradients and update encoder
    optimizer.zero_grad()
    loss.backward()    # backpropagation computes gradients for each encoder parameter
    optimizer.step()   # apply computed gradients to adjust encoder's internal weights

if __name__ == "__main__":
    main()
