import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from util.net import Net


def main():
    """
    Script used to pretrain the network for the drift-parking example. See Section VI.B of chung2025provably.

    References:
    1. Chung, Long Kiu, and Shreyas Kousik. "Provably-Safe Neural Network Training Using Hybrid Zonotope Reachability
       Analysis." arXiv preprint arXiv:2501.13023 (2025).

    Author: Long Kiu Chung
    Created: 2025/03/25
    Updated: 2025/05/01
    """

    # RNG seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Input domain
    x_lo = -0.1
    x_hi = 0.1
    y_lo = -0.1
    y_hi = 0.1

    # Output domain
    V_hi = 11.
    V_lo = 9.
    beta_hi = (2. / 9.) * np.pi
    beta_lo = np.pi / 6.
    normalize_factor = (V_hi - V_lo) / (beta_hi - beta_lo)

    # Setup network
    network_size = (2, 20, 2)
    net = Net(network_size)

    # Training features
    N_train = 10000
    X_train = np.random.uniform([x_lo, y_lo], [x_hi, y_hi], (N_train, 2))

    # Training labels
    Y_train = torch.hstack((torch.ones((N_train, 1)) * (V_hi + V_lo) * 0.5,
                            torch.ones((N_train, 1)) * (beta_hi + beta_lo) * 0.5 * normalize_factor))

    # Move to PyTorch
    device = torch.device("cpu")
    X_train = torch.as_tensor(X_train, dtype=torch.float).to(device)
    Y_train = torch.as_tensor(Y_train, dtype=torch.float).to(device)

    # Train the network
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    loss = nn.MSELoss()

    num_iters = 5000

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
    save_path = save_dir / f'prenet_ctrl_drift_{size_str}.pth'

    # Save the model
    torch.save(net.state_dict(), save_path)
    

if __name__ == "__main__":
    main()
