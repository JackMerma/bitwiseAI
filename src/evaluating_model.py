from training_source import *

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

xor_model, and_model, or_model, not_model = load_models("xor_model.tar", "and_model.tar", "or_model.tar", "not_model.tar")

# evaluation
xor_test_dataset = XORDataset(size=500)
and_test_dataset = ANDDataset(size=500)
or_test_dataset = ORDataset(size=500)
not_test_dataset = NOTDataset(size=500)

xor_test_data_loader = data.DataLoader(xor_test_dataset, batch_size=128, shuffle=False, drop_last=False)
and_test_data_loader = data.DataLoader(and_test_dataset, batch_size=128, shuffle=False, drop_last=False)
or_test_data_loader = data.DataLoader(or_test_dataset, batch_size=128, shuffle=False, drop_last=False)
not_test_data_loader = data.DataLoader(not_test_dataset, batch_size=128, shuffle=False, drop_last=False)

eval_model(xor_model, xor_test_data_loader, "XOR")
eval_model(and_model, and_test_data_loader, "AND")
eval_model(or_model, or_test_data_loader, "OR")
eval_model(not_model, not_test_data_loader, "NOT")
