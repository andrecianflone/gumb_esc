import torch
import torch.nn.functional as F
from torch.nn import functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

class GumbelSoftmax():
    def __init__(self, hard, categories, n_categories):
        super().__init__()
        self.categories = categories # Number of classes per distribution
        self.n_categories = n_categories # Of categorical distributions
        self.hard = hard

    def sample(self, logits, temperature):
        y = logits + sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def rsample(self, logits, temperature):
        """
        If hard, returns straight-through Gumbel-Softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.sample(logits, temperature)

        if not self.hard:
            return y.view(-1, self.n_categories * self.categories)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, self.n_categories * self.categories)


