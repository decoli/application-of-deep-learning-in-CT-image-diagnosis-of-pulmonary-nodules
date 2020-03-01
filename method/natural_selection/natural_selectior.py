import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-input', default=None)

    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='enables CUDA train')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed (default: 1)')

    args = parser.parse_args()

    # set device for pytorch
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    args.device = device

    # set random seed
    random.seed(args.seed)

    return args

class NaturalSelector(nn.Module):
    def __init__(self, args):
        super().__init__()

        # input 1 * 64 * 64

        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        ) # 32 * 32 * 32

        self.conv_2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        ) # 64 * 16 * 16

        self.conv_3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        ) # 128 * 8 * 8
    
        self.conv_4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        ) # 256 * 4 * 4

        self.fc_1 = nn.Linear(in_features=256 * 4 * 4, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=100)

    def forward(self, input_image)
        size_batch_in = input_image.size(0)

        out = self.conv_1(input_image)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        
        out = out.view(size_batch_in, -1)
        out = self.fc_1(out)
        out = self.fc_2(out)

        return out

def train():


def test():


if __name__ == '__main__':
    # get argument
    args = argument()



    # model instance
    model_natural_selector = NaturalSelector(args).to(args.device)x

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.rate_learning)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # model train and test
    for epoch in range(1, args.epoch + 1):
        train(model, model_vae, optimizer, criterion, train_loader, epoch, args, visdom)
        test(model, model_vae, test_loader, epoch, args, visdom)

        print('---epoch: {epoch:>4}---'.format(epoch=epoch))
