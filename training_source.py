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


class ANDDataset(Dataset):

    def __init__(self, size, std=0.1):
        super().__init__(size, std)
        self.generate_continuous_and()

    def generate_continuous_and(self):
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)

        # calculate AND output using sum
        label = (data.sum(dim=1) == 2).to(torch.long)
        
        # gaussian noise
        data += self.std * torch.randn(data.shape)

        # data & label
        self.data = data
        self.label = label


class ORDataset(Dataset):

    def __init__(self, size, std=0.1):
        super().__init__(size, std)
        self.generate_continuous_or()

    def generate_continuous_or(self):
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)

        # calculate OR output using sum
        label = (data.sum(dim=1) > 0).to(torch.long)
        
        # gaussian noise
        data += self.std * torch.randn(data.shape)

        # data & label
        self.data = data
        self.label = label


class NOTDataset(Dataset):

    def __init__(self, size, std=0.1):
        super().__init__(size, std)
        self.generate_continuous_not()

    def generate_continuous_not(self):
        data = torch.randint(low=0, high=2, size=(self.size, 1), dtype=torch.float32)

        # calculate OR output using sum
        label = (((data.sum(dim=1)) + 1) % 2).to(torch.long)
        
        # gaussian noise
        data += self.std * torch.randn(data.shape)

        # data & label
        self.data = data
        self.label = label


def train_model(model, optimizer, data_loader, loss_module, num_epochs=100, name_model="Operation"):
    model.train()

    for epoch in tqdm(range(num_epochs), desc=f"{name_model} model"):
        for data_inputs, data_labels in data_loader:

            # step 1: move input data to device (just if we use GPU)
            # data_inputs = data_inputs.to(device)
            # data_labels = data_labels.to(device)

            # step 2: run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)

            # step 3: calculate the loss
            loss = loss_module(preds, data_labels.float())

            # step 4: perform backpropagation
            optimizer.zero_grad()
            loss.backward()

            # step 5: update parameters
            optimizer.step()

def load_models(xor_model_name, and_model_name, or_model_name, not_model_name):
    # loading model
    xor_static_dict = torch.load(f"models/{xor_model_name}")
    and_static_dict = torch.load(f"models/{and_model_name}")
    or_static_dict = torch.load(f"models/{or_model_name}")
    not_static_dict = torch.load(f"models/{not_model_name}")

    xor_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    and_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    or_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    not_model = SimpleClassifier(num_inputs=1, num_hidden=2, num_outputs=1)

    xor_model.load_state_dict(xor_static_dict)
    and_model.load_state_dict(and_static_dict)
    or_model.load_state_dict(or_static_dict)
    not_model.load_state_dict(not_static_dict)

    return xor_model, and_model, or_model, not_model

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
