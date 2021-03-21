""""
Gumbel Esc
Some code based on Gumbel Softmax from:
Eric Jang: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
Devinder Kumar
Yongfei Yan: https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
"""

import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from distributions import GumbelSoftmax, GumbelEscort

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--temp', type=float, default=1.0, metavar='S',
                    help='tau(temperature) (default: 1.0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hard', action='store_true', default=False,
                    help='hard Gumbel softmax')
parser.add_argument('--distribution', default='softmax', nargs=1,
                    choices=['softmax', 'gumbel'],
                    help='Which type of relaxed Gumbel-Max (default: %(default)s)')

parser.add_argument('--gumbel_escort_p', type=int, default=2, metavar='N',
                    help='Value of p for the Escort dist. (default: 2)')
parser.add_argument('--latent_dim', type=float, default=30, metavar='S',
                    help='Gumbel latent dim (default: 1.0)')
parser.add_argument('--categorical_dim', type=int, default=10, metavar='N',
                    help='Number of classes per Categorical (default: 10)')
parser.add_argument('--temp_min', type=float, default=0.5, metavar='S',
                    help='Min temp (default: 0.5)')
parser.add_argument('--ANNEAL_RATE', type=float, default=0.00003, metavar='S',
                    help='Anneal rate (default: 0.00003)')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True}
else:
    kwargs = {}

class VAE(nn.Module):
    def __init__(self, latent_distribution, latent_dim, categorical_dim):
        super().__init__()
        self.latent_dist = latent_distribution
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, x, temp):
        q = self.encode(x.view(-1, 784))
        q_y = q.view(q.size(0), self.latent_dim, self.categorical_dim)
        z = self.latent_dist.rsample(q_y, temp)
        return self.decode(z), F.softmax(q_y, dim=-1).reshape(*q.size())

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False) / x.shape[0]

    log_ratio = torch.log(qy * args.categorical_dim + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    temp = args.temp
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, qy = model(data, temp)
        loss = loss_function(recon_batch, data, qy)
        loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()
        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-args.ANNEAL_RATE * batch_idx), args.temp_min)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    temp = args.temp
    for i, (data, _) in enumerate(test_loader):
        data = data.to(device)
        recon_batch, qy = model(data, temp)
        test_loss += loss_function(recon_batch, data, qy).item() * len(data)
        if i % 100 == 1:
            temp = np.maximum(temp * np.exp(-args.ANNEAL_RATE * i), args.temp_min)
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def main(args):
    if args.distribution == 'softmax':
        latent_dist = GumbelSoftmax(hard, args.categorical_dim, args.latent_dim)
    elif args.distribution == 'escort':
        latent_dist = GumbelEscort(hard, args.categorical_dim, args.latent_dim, p=2)

    model = VAE(args.hard, args.latent_dim, args.categorical_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=False,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

        M = 64 * args.latent_dim
        np_y = np.zeros((M, args.categorical_dim), dtype=np.float32)
        np_y[range(M), np.random.choice(args.categorical_dim, M)] = 1
        np_y = np.reshape(np_y, [M // args.latent_dim, args.latent_dim, args.categorical_dim])
        sample = torch.from_numpy(np_y).view(M // args.latent_dim, args.latent_dim * args.categorical_dim).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.data.view(M // args.latent_dim, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')


if __name__ == '__main__':
    args = parser.parse_args()
    args.device = device
    main(args)
