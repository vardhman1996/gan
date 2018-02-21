import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

BATCH_SIZE = 32
k = 1
d_learning_rate = 2e-4
g_learning_rate = 2e-4

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)


def get_generator_noise_input():
    return lambda m, w, h: torch.rand(m, w, h)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lin1 = nn.Linear(784, 784)
        self.lin2 = nn.Linear(784, 784)
        # self.scale = nn.Parameter(torch.FloatTensor([255]))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.sigmoid(self.lin2(x))
        return torch.mul(x, 255.0).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.fn = nn.Linear(28 * 28 * 5, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.fn(x.view(x.size(0), -1))
        return F.sigmoid(x)


g_sampler = get_generator_noise_input()
G = Generator()
D = Discriminator()
d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=0.9)
g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate, momentum=0.9)

def generative_loss(f_input):
    # (1/m) sum_1_m log(1-D(G(z)))
    batch_loss = torch.log(torch.add(torch.neg(D(G(f_input))), 1.0))
    print "G-Loss:", torch.div(torch.sum(batch_loss), f_input.size(0))
    return torch.div(torch.sum(batch_loss), f_input.size(0))

def discriminative_loss(r_input, f_input):
    # (1/m) sum_1_m log(D(x) + log(1-D(G(z)))
    d_loss = torch.log(D(r_input))
    g_loss = torch.log(torch.add(torch.neg(D(G(f_input))), 1.0))
    batch_loss = torch.add(d_loss, g_loss)
    print "D-Loss:", torch.div(torch.sum(batch_loss), r_input.size(0))
    return torch.div(torch.sum(batch_loss), r_input.size(0))


def train_discriminative(data):
    D.train()
    G.eval()
    r_data = Variable(data)
    f_data = Variable(torch.randn(data.size(0), 1, data.size(2), data.size(3)))
    d_optimizer.zero_grad()
    d_loss = discriminative_loss(r_data, f_data)
    d_loss.backward()
    d_optimizer.step()

def train_generative(data):
    G.train()
    D.eval()
    f_data = Variable(torch.randn(data.size(0), 1, data.size(2), data.size(3)))
    g_optimizer.zero_grad()
    g_loss = generative_loss(f_data)
    g_loss.backward()
    g_optimizer.step()


def main(steps):
    for _ in range(steps):
        for batch_num, (data, _) in enumerate(train_loader):
            train_discriminative(data)
            if (batch_num + 1) % k == 0:
                train_generative(data)


if __name__ == "__main__":
    main(4)