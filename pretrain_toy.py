import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from util.net import Net


def main():
    """
    Script used to pretrain the network for the toy example. See Section V of chung2025provably.

    References:
    1. Chung, Long Kiu, and Shreyas Kousik. "Provably-Safe Neural Network Training Using Hybrid Zonotope Reachability
       Analysis." arXiv preprint arXiv:2501.13023 (2025).

    Author: Long Kiu Chung
    Created: 2025/01/14
    Updated: 2025/04/30
    """

    # User parameters
    network_size = (10,)
    # network_size = (20,)
    # network_size = (30,)
    # network_size = (60,)
    # network_size = (120,)
    # network_size = (240,)
    # network_size = (120, 120)
    # network_size = (80, 80, 80)

    # RNG seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Setup network
    network_size = (2,) + network_size + (2,)

    net = Net(network_size)

    # Training features
    N_train = 10000
    X_train = np.random.uniform(-1, 1, (N_train, 2))

    # Training labels
    Y_train = f_nonl(X_train)

    # Move to PyTorch
    device = torch.device("cpu")
    X_train = torch.as_tensor(X_train, dtype=torch.float).to(device)
    Y_train = torch.as_tensor(Y_train, dtype=torch.float).to(device)

    # Train the network
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    loss = nn.MSELoss()

    num_iters = 500

    # training loop:
    for i in range(num_iters):
        optimizer.zero_grad()  # zero the gradient buffers

        pred = net(X_train)

        output = loss(pred, Y_train)
        output.backward()
        optimizer.step()

    print("Final Loss: ", loss(net(X_train), Y_train))

    # Specify the save directory
    current_dir = Path(__file__).parent.resolve()
    save_dir = current_dir / 'data' / 'network'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create the file name
    size_str = ''.join(str(n) for n in network_size)
    save_path = save_dir / f'prenet_toy_{size_str}.pth'

    # Save the model
    torch.save(net.state_dict(), save_path)


def f_nonl(x):
    """
    Toy nonlinear function to compare with chung2021constrained.

    References:
    1. Chung, Long Kiu, et al. "Constrained feedforward neural network training via reachability analysis." arXiv
       preprint arXiv:2107.07696 (2021).

    INPUTS:
    x: numpy array, size n_train*2; input of the toy function

    OUTPUTS:
    y: numpy array, size n_train*2; output of the toy function

    Author: Long Kiu Chung
    Created: 2025/01/14
    Updated: 2025/04/30
    """

    return np.array([x[:, 0]**2 + np.sin(x[:, 1]), x[:, 1]**2 + np.sin(x[:, 0])]).T


if __name__ == "__main__":
    main()
