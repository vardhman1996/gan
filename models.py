import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 128
CUDA = True
k = 1
d_learning_rate = 0.0002
g_learning_rate = 0.000002
F_DIM = 256
OUT_DIM = 784

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lin1 = nn.Linear(F_DIM, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 512)
        self.lin4 = nn.Linear(512, OUT_DIM)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.lin1(x), 0.2)
        x = F.dropout(x, p=0.5)
        x = F.leaky_relu(self.lin2(x), 0.2)
        x = F.dropout(x, p=0.5)
        x = F.leaky_relu(self.lin3(x), 0.2)
        x = F.dropout(x, p=0.5)
        x = F.tanh(self.lin4(x))
        return x.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(OUT_DIM, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, 1)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.lin1(x), 0.2)
        x = F.leaky_relu(self.lin2(x), 0.2)
        x = F.leaky_relu(self.lin3(x), 0.2)
        x = self.lin4(x)
        x = F.sigmoid(x)
        return x


G = Generator()
D = Discriminator()

if CUDA:
    G.cuda()
    D.cuda()

BCELoss = nn.BCELoss()
d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=0.9)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)

def generative_loss(f_data):
    dg_ = D(G(f_data))
    y_ = torch.ones(f_data.size()[0], 1)
    if CUDA:
        y_ = y_.cuda()
    g_loss = BCELoss(dg_, Variable(y_))
    return g_loss


def discriminative_loss(r_data, f_data):
    d, dg_ = D(r_data), D(G(f_data))
    y, y_ = torch.ones(r_data.size()[0], 1), torch.zeros(f_data.size()[0], 1)
    if CUDA:
        y = y.cuda()
        y_ = y_.cuda()
    d_loss = BCELoss(d, Variable(y)) + BCELoss(dg_, Variable(y_))
    return d_loss


def train_discriminative(data):
    d_optimizer.zero_grad()
    r_data = Variable(data)
    z = torch.randn(data.size(0), 1, F_DIM)
    if CUDA:
        z = z.cuda()
    f_data = Variable(z)
    d_loss = discriminative_loss(r_data, f_data)
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data[0]


def train_generative(data):
    g_optimizer.zero_grad()
    z = torch.randn(data.size(0), 1, F_DIM)
    if CUDA:
        z = z.cuda()
    f_data = Variable(z)
    g_loss = generative_loss(f_data)
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data[0]


def get_image():
    z = torch.randn(1, 1, F_DIM)
    if CUDA:
        z = z.cuda()
    f_data = Variable(z)
    img = F.sigmoid(G(f_data)).cpu()
    img = img.data.numpy()
    img = img.reshape(28,28)
    return img


def plot_loss(d_loss, g_loss, step):
    fig, ax = plt.subplots()
    ax.plot(np.arange(d_loss.shape[0]), d_loss)
    ax.plot(np.arange(g_loss.shape[0]), g_loss)
    ax.set_xlabel("Minibatch step")
    ax.set_ylabel("Loss")
    ax.legend(['d_loss', 'g_loss'])
    fig.savefig('plots/loss_{0}'.format(step))
    plt.close()


def plot_image(img, step):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    fig.savefig('plots/generative_img_{0}'.format(step))
    plt.close()


def main(steps):
    d_loss = []
    g_loss = []
    for step in range(steps):
        dl = 0.0
        gl = 0.0
        for batch_num, (data, _) in enumerate(train_loader):
            if CUDA:
                data = data.cuda()
            dl = train_discriminative(data)
            if (batch_num + 1) % k == 0:
                gl = train_generative(data)
        d_loss.append(dl)
        g_loss.append(gl)
        print "Epoch {0}, d_loss {1}, g_loss {2}".format(step, d_loss[-1], g_loss[-1])
        img = get_image()
        plot_image(img, step)
        plot_loss(np.array(d_loss), np.array(g_loss), step)


if __name__ == "__main__":
    main(500)