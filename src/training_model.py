from training_source import *
from plot import *

torch.manual_seed(17)

# creating model
model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)

# creating XOR dataset
xor_dataset = XORDataset(size=100)

# plotting
visualize_samples(xor_dataset.data, xor_dataset.label, file_name="xor_dataset.png")
