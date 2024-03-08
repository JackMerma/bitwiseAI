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


def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    model.train()

    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:

            # step 1: mode input data to device (just if we use GPU)
            # data_inputs = data_inputs.to(device)
            # data_labels = data_labels.to(device)

            # step 2: run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)

            # step 3: calculate the loss
            loss = loss_module(preds, data_labels.float())

            # step 4: preform backpropagation
            optimizer.zero_grad()
            loss.backward()

            # step 5: update parameters
            optimizer.step()

def eval_model(model, data_loader, name_model):
    model.eval()
    true_preds, num_preds = 0., 0.

    with torch.no_grad():
        for data_inputs, data_labels in data_loader:
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds)
            preds_labels = (preds >= 0.5).long()

            true_preds += (preds_labels == data_labels).sum()
            num_preds += data_labels.shape[0]
    
    acc = true_preds / num_preds
    print(f"Accuracy of the {name_model} model: {100.0*acc:4.2f}%")
