import torch
import matplotlib.pyplot as plt

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
