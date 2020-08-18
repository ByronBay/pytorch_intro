import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

# Import mnist dataset from cvs file and convert it to torch tensor

with open('../data_temp/mnist_train.csv', 'r') as f:
    mnist_train = f.readlines()

# Images
X_train = np.array([[float(j) for j in i.strip().split(',')][1:] for i in mnist_train])
X_train = X_train.reshape((-1, 1, 28, 28))
X_train = torch.tensor(X_train)

# Labels
y_train = np.array([int(i[0]) for i in mnist_train])
y_train = y_train.reshape(y_train.shape[0], 1)
y_train = torch.tensor(y_train)

del mnist_train


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def imshow(img, title=''):
    """Plot the image batch.
    """
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)), cmap='gray')
    plt.show()


# Dataset w/o any tranformations
train_dataset_normal = CustomTensorDataset(tensors=(X_train, y_train), transform=None)
train_loader = torch.utils.data.DataLoader(train_dataset_normal, batch_size=9)

# iterate
for i, data in enumerate(train_loader):
    x, y = data
    imshow(torchvision.utils.make_grid(x, 4), title='Normal')
    break  # we need just one batch


# Let's add some transforms

# Dataset with flipping tranformations

def vflip(tensor):
    """Flips tensor vertically.
    """
    tensor = tensor.flip(1)
    return tensor


def hflip(tensor):
    """Flips tensor horizontally.
    """
    tensor = tensor.flip(2)
    return tensor


train_dataset_vf = CustomTensorDataset(tensors=(X_train, y_train), transform=vflip)
train_loader = torch.utils.data.DataLoader(train_dataset_vf, batch_size=16)

result = []

for i, data in enumerate(train_loader):
    x, y = data
    imshow(torchvision.utils.make_grid(x, 4), title='Vertical flip')
    break

train_dataset_hf = CustomTensorDataset(tensors=(X_train, y_train), transform=hflip)
train_loader = torch.utils.data.DataLoader(train_dataset_hf, batch_size=16)

result = []

for i, data in enumerate(train_loader):
    x, y = data
    imshow(torchvision.utils.make_grid(x, 4), title='Horizontal flip')
    break
