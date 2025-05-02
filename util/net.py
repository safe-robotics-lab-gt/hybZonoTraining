import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Class for neural networks fully-connected with ReLU activation functions.

    Author: Long Kiu Chung
    Created: 2025/01/14
    Updated: 2025/05/01
    """

    def __init__(self, network_size):
        """
        Constructor for the ReLU neural network.

        INPUTS:
        network_size: tuple of int; denotes the input size, widths, and output size of the network

        OUTPUTS:
        N/A

        Created: 2025/01/14
        Updated: 2025/04/30
        """

        super(Net, self).__init__()

        self.layers = nn.ModuleList()
        for n_in, n_out in zip(network_size[:-1], network_size[1:]):
            self.layers.append(nn.Linear(n_in, n_out))

    def forward(self, x):
        """
        Forward pass of the neural network.

        INPUTS:
        x: torch tensor, size n_train*n_in; input to the network

        OUTPUTS:
        x: torch tensor, size n_train*n_out; output of the network

        Created: 2025/01/14
        Updated: 2025/04/30
        """

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)

        return x
