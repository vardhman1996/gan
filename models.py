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
k = 2
d_learning_rate = 0.0002
g_learning_rate = 0.000002
F_DIM = 500
OUT_DIM = 784

# kwargs = {'num_workers': , 'pin_memory': True} if CUDA else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('./data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        # transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=BATCH_SIZE, shuffle=True, **kwargs)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lin1 = nn.Linear(F_DIM, 256)
        self.lin2 = nn.Linear(256, 512)
        self.lin3 = nn.Linear(512, 1024)
        self.lin4 = nn.Linear(1024, OUT_DIM)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.sigmoid(self.lin4(x))
        return x.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.lin1 = nn.Linear(784, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(256, 1)
        # self.fn = nn.Linear(28 * 28 * 5, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, 0.3)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, 0.3)
        x = F.relu(self.lin3(x))
        x = F.dropout(x, 0.3)
        x = self.lin4(x)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        return F.sigmoid(x)


G = Generator()
D = Discriminator()

if CUDA:
    G.cuda()
    D.cuda()

BCELoss = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)
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
    G.eval()
    r_data = Variable(data)
    z = torch.rand(data.size(0), 1, F_DIM)
    if CUDA:
        z = z.cuda()
    f_data = Variable(z)
    d_optimizer.zero_grad()
    d_loss = discriminative_loss(r_data, f_data)
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data[0]


def train_generative(data):
    D.eval()
    z = torch.rand(data.size(0), 1, F_DIM)
    if CUDA:
        z = z.cuda()
    f_data = Variable(z)
    g_optimizer.zero_grad()
    g_loss = generative_loss(f_data)
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data[0]


def get_image():
    G.eval()
    z = torch.rand(1, 1, F_DIM)
    if CUDA:
        z = z.cuda()
    f_data = Variable(z)
    img = G(f_data).cpu()
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
            # d_loss.append(dl)
            if (batch_num + 1) % k == 0:
                gl = train_generative(data)
                # g_loss.append(gl)
        d_loss.append(dl)
        g_loss.append(gl)
        print "Epoch {0}, d_loss {1}, g_loss {2}".format(step, d_loss[-1], g_loss[-1])
        img = get_image()
        plot_image(img, step)
        plot_loss(np.array(d_loss), np.array(g_loss), step)


torch.manual_seed(1)
if CUDA:
    torch.cuda.manual_seed(1)


if __name__ == "__main__":
    main(100)