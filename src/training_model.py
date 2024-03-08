from training_source import *
from plot import *

torch.manual_seed(17)

# creating model
xor_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)

# creating XOR dataset
xor_dataset = XORDataset(size=2500)

# plotting
visualize_samples(xor_dataset.data, xor_dataset.label, file_name="xor_dataset.png")

# data loader
xor_data_loader = data.DataLoader(xor_dataset, batch_size=150, shuffle=True)

# loss module & optimizer
loss_module = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(xor_model.parameters(), lr=0.1)

# training model
train_model(xor_model, optimizer, xor_data_loader, loss_module)

# saving model
state_dict = xor_model.state_dict()
torch.save(state_dict, "models/xor_model.tar")
