from __future__ import print_function

import argparse
import os
import sys

import pandas as pd
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

#d append sys.path
sys.path.append(os.getcwd())
from pre_processing.utility import get_image_info, get_coordinate

# define argument
def argument():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--path-input', default=None)
    parser.add_argument('--dir-image', type=str)
    parser.add_argument('--size-cutting', default=28)

    parser.add_argument('--rate-train', default=0.9, type=float)
    parser.add_argument('--size-batch', type=int, default=128, metavar='N',
                        help='input batch size for train (default: 128)')
    parser.add_argument('--epoch', type=int, default=10, metavar='N',
                        help='number of epoch to train (default: 10)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA train')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging train status')

    parser.add_argument('--num-cross', default=None, type=int)
    parser.add_argument('--use-cross', default=None, type=int)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    return args, device, kwargs

# define data set
class DatasetTrain():
    def __init__(self, list_data_set):
        self.list_data_set

    def __len__(self):
        return self.list_data_set

    def __getitem__(self, idx):
        current_item = self.list_data[idx]
        coordinate = get_coordinate(current_item)

        # get image path
        path_image = 

class DatasetTest():
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass

# define class 
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(int(args.size_cutting * args.size_cutting), 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, int(args.size_cutting * args.size_cutting))

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(model, train_loader, epoch, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(model, test_loader, epoch, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, args.size_cutting, args.size_cutting)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    # get argument
    args, device, kwargs = argument()

    # get image info
    info_luna16 = pd.read_csv(args.path_info, index_col=0)
    list_info_image = get_image_info(info_luna16)

    len_list_info_image = len(list_info_image)
    list_train = list_info_image[: int(len_list_info_image * args.rate_train)]
    list_test = list_info_image[int(len_list_info_image * args.rate_train): ]

    # define date loader
    data_set_train = DatasetTrain(list_train)
    data_set_test = DatasetTest(list_test)
    train_loader = torch.utils.data.DataLoader(data_set_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_set_test, shuffle=True)

    # model instance
    model = VAE(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epoch + 1):
        train(model, train_loader, epoch, device)
        test(model, test_loader, epoch, device)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, int(args.size_cutting), int(args.size_cutting)),
                       'results/sample_' + str(epoch) + '.png')
