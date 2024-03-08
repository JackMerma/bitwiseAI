import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm.notebook import tqdm


class SimpleClassifier(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


class Dataset(data.Dataset):

    def __init__(self, size, std=0.1):
        super().__init__()
        self.size = size
        self.std = std

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class XORDataset(Dataset):

    def __init__(self, size, std=0.1):
        super().__init__(size, std)
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)

        # calculate XOR output using sum
        label = (data.sum(dim=1) == 1).to(torch.long)
        
        # gaussian noise
        data += self.std * torch.randn(data.shape)

        # data & label
        self.data = data
        self.label = label

