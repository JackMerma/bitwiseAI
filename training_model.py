from training_source import *
from plot import *

torch.manual_seed(17)

# creating model
xor_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
and_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
or_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
not_model = SimpleClassifier(num_inputs=1, num_hidden=2, num_outputs=1)

# creating XOR dataset
xor_dataset = XORDataset(size=2500)
and_dataset = ANDDataset(size=2500)
or_dataset = ORDataset(size=2500)
not_dataset = NOTDataset(size=2500)

# plotting
visualize_samples(xor_dataset.data, xor_dataset.label, title="XOR Dataset", file_name="xor_dataset.png")
visualize_samples(and_dataset.data, and_dataset.label, title="AND Dataset", file_name="and_dataset.png")
visualize_samples(or_dataset.data, or_dataset.label, title="OR Dataset", file_name="or_dataset.png")
visualize_samples(not_dataset.data, not_dataset.label, title="NOT Dataset", file_name="not_dataset.png")

# data loader
xor_data_loader = data.DataLoader(xor_dataset, batch_size=150, shuffle=True)
and_data_loader = data.DataLoader(and_dataset, batch_size=150, shuffle=True)
or_data_loader = data.DataLoader(or_dataset, batch_size=150, shuffle=True)
not_data_loader = data.DataLoader(not_dataset, batch_size=150, shuffle=True)

# loss module & optimizer
xor_loss_module = nn.BCEWithLogitsLoss()
xor_optimizer = torch.optim.SGD(xor_model.parameters(), lr=0.1)
and_loss_module = nn.BCEWithLogitsLoss()
and_optimizer = torch.optim.SGD(and_model.parameters(), lr=0.1)
or_loss_module = nn.BCEWithLogitsLoss()
or_optimizer = torch.optim.SGD(or_model.parameters(), lr=0.1)
not_loss_module = nn.BCEWithLogitsLoss()
not_optimizer = torch.optim.SGD(not_model.parameters(), lr=0.1)

# training model
train_model(xor_model, xor_optimizer, xor_data_loader, xor_loss_module, name_model="XOR")
train_model(and_model, and_optimizer, and_data_loader, and_loss_module, name_model="AND")
train_model(or_model, or_optimizer, or_data_loader, or_loss_module, name_model="OR")
train_model(not_model, not_optimizer, not_data_loader, not_loss_module, name_model="NOT")

# saving model
xor_state_dict = xor_model.state_dict()
and_state_dict = and_model.state_dict()
or_state_dict = or_model.state_dict()
not_state_dict = not_model.state_dict()
torch.save(xor_state_dict, "models/xor_model.tar")
torch.save(and_state_dict, "models/and_model.tar")
torch.save(or_state_dict, "models/or_model.tar")
torch.save(not_state_dict, "models/not_model.tar")

# visualizing classification
visualize_classification(xor_model, xor_dataset.data, xor_dataset.label, title="XOR Data classification", file_name="xor_classification")
visualize_classification(and_model, and_dataset.data, and_dataset.label, title="AND Data classification", file_name="and_classification")
visualize_classification(or_model, or_dataset.data, or_dataset.label, title="OR Data classification", file_name="or_classification")
visualize_classification(not_model, not_dataset.data, not_dataset.label, title="NOT Data classification", file_name="not_classification")
