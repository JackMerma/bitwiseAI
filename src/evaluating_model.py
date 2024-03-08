from training_source import *
from evaluating_model import *

# loading model
xor_static_dict = torch.load("models/xor_model.tar")
xor_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
xor_model.load_state_dict(xor_static_dict)

# evaluation
xor_test_dataset = XORDataset(size=500)
xor_test_data_loader = data.DataLoader(xor_test_dataset, batch_size=128, shuffle=False, drop_last=False)
eval_model(xor_model, xor_test_data_loader, "XOR")
