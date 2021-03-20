import torch
import torch.nn.functional as F
from torch.nn import functional as F

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

class GumbelSoftmax():
    def __init__(self, temperature, hard):
        super().__init__():
        self.temperature = temperature
        self.hard = hard

    def sample(logits, temperature):
        y = logits + sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def rsample(logits):
        """
        If hard, returns straight-through Gumbel-Softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = gumbel_softmax_sample(logits, self.temperature)

        if not self.hard:
            return y.view(-1, latent_dim * categorical_dim)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, latent_dim * categorical_dim)

