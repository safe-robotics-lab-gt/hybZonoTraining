import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from util.net import Net


def main():
    """
    Script used to pretrain the network for the forward-invariant double integrator example. See Section VI.A of
    chung2025provably.

    References:
    1. Chung, Long Kiu, and Shreyas Kousik. "Provably-Safe Neural Network Training Using Hybrid Zonotope Reachability
       Analysis." arXiv preprint arXiv:2501.13023 (2025).

    Author: Long Kiu Chung
    Created: 2025/03/03
    Updated: 2025/05/01
    """

    # RNG seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Double integrator dynamics
    x_lo = -2.
    x_hi = 2.
    v_lo = -2.
    v_hi = 2.

    # Setup network
    network_size = (2, 3, 1)
    net = Net(network_size)

    # Training features
    N_train = 10000
    X_train = np.random.uniform([x_lo, v_lo], [x_hi, v_hi], (N_train, 2))

    # Training labels
    Y_train = u_double(X_train)

    # Move to PyTorch
    device = torch.device("cpu")
    X_train = torch.as_tensor(X_train, dtype=torch.float).to(device)
    Y_train = torch.as_tensor(Y_train, dtype=torch.float).to(device)

    # Train the network
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    loss = nn.MSELoss()

    num_iters = 900

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
    save_path = save_dir / f'prenet_ctrl_doubleint_{size_str}.pth'

    # Save the model
    torch.save(net.state_dict(), save_path)


def u_double(s_in):
    """
    Non-forward-invariant controller for double integrator.

    INPUTS:
    s_in: numpy array, size n_train*2; first column is position, second column is velocity

    OUTPUTS:
    u_in: numpy array, size n_train*1; control (acceleration) of the double integrator

    Author: Long Kiu Chung
    Created: 2025/03/03
    Updated: 2025/05/01
    """

    # s_in is defined as [x_in, v_in]
    # Extract parameters from s_in
    x_in = s_in[:, 0]
    v_in = s_in[:, 1]

    # Braking controller
    a_in = -2.*x_in - 1.*v_in  # Define a deliberately bad controller
    u_in = np.array([a_in])

    return u_in.T


if __name__ == "__main__":
    main()
