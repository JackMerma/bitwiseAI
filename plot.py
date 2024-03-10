import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

def visualize_samples(data, label, file_name="fig.png"):

    r, c = data.shape

    if c == 1:
        data = torch.cat((data, torch.ones((r, 1), dtype=data.dtype)), dim=1)

    data0 = data[label == 0]
    data1 = data[label == 1]

    #plotting
    plt.scatter(data0[:, 0], data0[:, 1], edgecolor="#333", label="class 0")
    plt.scatter(data1[:, 0], data1[:, 1], edgecolor="#333", label="class 1")
    plt.title("Dataset")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()
    plt.savefig(f"plots/{file_name}")

@torch.no_grad()
def visualize_classification(model, data, label, file_name="fig.png"):
    
    r, c = data.shape

    if c == 1:
        data = torch.cat((data, (0.5 * torch.ones((r, 1), dtype=data.dtype))), dim=1)

    data0 = data[label == 0]
    data1 = data[label == 1]

    plt.scatter(data0[:, 0], data0[:, 1], edgecolor="#333", label="class 0")
    plt.scatter(data1[:, 0], data1[:, 1], edgecolor="#333", label="class 1")
    plt.title("Dataset")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

    c0 = torch.Tensor(to_rgba("C0"))
    c1 = torch.Tensor(to_rgba("C1"))
    x1 = torch.arange(-0.5, 1.5, step=0.01)
    x2 = torch.arange(-0.5, 1.5, step=0.01)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='ij')
    model_inputs = torch.stack([xx1, xx2], dim=-1)

    if c == 1:
        model_inputs = torch.stack([xx2], dim=-1)

    print("model_inputs: ", model_inputs)
    preds = model(model_inputs)
    preds = torch.sigmoid(preds)
    output_image = (1 - preds) * c0[None, None] + preds * c1[None, None]
    output_image = output_image.cpu().numpy()

    plt.imshow(output_image, origin='lower', extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)
    plt.savefig(f"plots/{file_name}")
