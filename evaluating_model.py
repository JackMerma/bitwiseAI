from training_source import *

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
