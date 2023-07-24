import torch
import random
from torch import nn

class ParticleMask(nn.Module):
    def __init__(self, group_size=4):
        super(ParticleMask, self).__init__()
        self.group_size = group_size

    def forward(self, x):
        assert x.dim() == 3, "Input tensor must be 3-dimensional (batch_size, seq_len, dim_in)"
        batch, seq_len, features = x.size()
        assert features == self.group_size, "Sequence length must be divisible by group_size"

        # Generate a mask tensor with the same shape as the input tensor
        mask = torch.ones(batch, seq_len, features, device=x.device)

        # Generate a random starting index for each sample in the batch and mask those features
        for b in range(batch):
            idx = random.randint(0, seq_len - 1)
            # idx = random.randint(0, 2) # leptons and MET only
            mask[b, idx, :] = 0

        return x * mask

class SpecificParticleMask(nn.Module):
    def __init__(self, group_size=4, particle=0):
        super(SpecificParticleMask, self).__init__()
        self.group_size = group_size
        self.particle = particle

    def forward(self, x):
        assert x.dim() == 3, "Input tensor must be 3-dimensional (batch_size, seq_len, dim_in)"
        batch, seq_len, features = x.size()

        # Generate a mask tensor with the same shape as the input tensor
        mask = torch.ones(batch, seq_len, features, device=x.device)

        # Generate a random starting index for each sample in the batch
        for b in range(batch):
            mask[b, self.particle, :] = 0

        return x * mask