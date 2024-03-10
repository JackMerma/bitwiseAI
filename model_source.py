import torch
from training_source import load_models

xor_model, and_model, or_model, not_model = load_models("xor_model.tar", "and_model.tar", "or_model.tar", "not_model.tar")

def evaluate_operation(model, data):
    model.eval()
    with torch.no_grad():
        data = data.unsqueeze(0)
        pred = model(data)
        pred = pred.squeeze(dim=1)
        return torch.sigmoid(pred)
        #return 1 if torch.sigmoid(pred) >= 0.5 else 0

def XOR(input1, input2):
    data = torch.tensor([input1, input2], dtype=torch.float32)
    return evaluate_operation(xor_model, data)

def AND(input1, input2):
    data = torch.tensor([input1, input2], dtype=torch.float32)
    return evaluate_operation(and_model, data)

def OR(input1, input2):
    data = torch.tensor([input1, input2], dtype=torch.float32)
    return evaluate_operation(or_model, data)

def NOT(input1):
    data = torch.tensor([input1], dtype=torch.float32)
    return evaluate_operation(not_model, data)

def IMP(input1, input2):
    return OR(NOT(input1), input2)

def BIC(input1, input2):
    return XOR(IMP(input1, input2), IMP(input2, input1))
