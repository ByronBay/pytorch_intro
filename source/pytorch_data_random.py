import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import pandas as pd


class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.labels = pd.get_dummies(self.data['emotion']).as_matrix()
        self.height = 28
        self.width = 28
        self.transform = transform

    def __getitem__(self, index):
        # This method should return only 1 sample and label
        # (according to "index"), not the whole dataset
        # So probably something like this for you:
        pixel_sequence = self.data['pixels'][index]
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(self.width, self.height)
        face = cv2.resize(face.astype('uint8'), (self.width, self.height))
        label = self.labels[index]

        return face, label

    def __len__(self):
        return len(self.labels)


dataset = CustomDatasetFromCSV('mnist_train.csv')
batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

# Usage Example:
num_epochs = 10
for epoch in range(num_epochs):
    # Train:
    for batch_index, (faces, labels) in enumerate(train_loader):
        pass