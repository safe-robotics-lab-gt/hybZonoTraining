import numpy as np
import os
import torch
import torch.optim as optim
from util.MILPLoss import shrinkabilityLoss
from util.hybZono import HybZono
from util.net import Net


def main():
    """
    Main script for the double integrator forward invariance example. See Section VI.A of chung2025provably.

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
    dt = 0.1  # Step size

    # Setup network
    network_size = (2, 3, 1)
    net = Net(network_size)

    # Read in the files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    X_now = HybZono.load(os.path.join(script_dir, 'data', 'input', 'input_doubleint.pt'))
    U_real = HybZono.load(os.path.join(script_dir, 'data', 'obstacle', 'obstacle_real_doubleint.pt'))
    U_check = HybZono.load(os.path.join(script_dir, 'data', 'obstacle', 'obstacle_check_doubleint.pt'))
    net.load_state_dict(torch.load(os.path.join(script_dir, 'data', 'network', f'prenet_ctrl_doubleint_231.pth')))

    # Make sure training is in CPU
    device = torch.device("cpu")
    net.to(device)

    # Constraint optimizer
    con_opt = optim.Adam(net.parameters(), lr=0.001)

    # Training loop
    iter_max = 700
    iter = 0
    shrinkDims = [0, 5, 6, 7, 8]  # n_r in chung2025provably

    while True:
        iter += 1

        # Zero the gradient
        con_opt.zero_grad()

        # Compute forward reachable set
        X_next = X_now.forwardReLUGraph(net)
        R = torch.tensor([[1., dt, 0.], [0., 1., dt]])
        X_next = X_next.affineMap(R)

        # Check collision with MILP
        if iter % 25 == 0:
            print("Iteration: ", iter)
            Q_check = X_next.intersect(U_check)
            isEmpty = Q_check.isEmpty()
            if isEmpty:
                print("Training successful!")
                break
        elif iter == iter_max:
            print("Max iteration reached!")
            break

        Q_real = X_next.intersect(U_real)
        loss = shrinkabilityLoss(Q_real, shrinkDims=shrinkDims)
        loss.backward()
        con_opt.step()


if __name__ == "__main__":
    main()
