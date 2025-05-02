import numpy as np
import os
import torch
import torch.optim as optim
from util.MILPLoss import shrinkabilityLoss
from util.hybZono import HybZono
from util.net import Net


def main():
    """
    Main script for the toy example. See Section V of chung2025provably.

    References:
    1. Chung, Long Kiu, and Shreyas Kousik. "Provably-Safe Neural Network Training Using Hybrid Zonotope Reachability
       Analysis." arXiv preprint arXiv:2501.13023 (2025).

    Author: Long Kiu Chung
    Created: 2025/01/14
    Updated: 2025/05/01
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

    # Read in the files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    Z = HybZono.load(os.path.join(script_dir, 'data', 'input', 'input_toy.pt'))
    U = HybZono.load(os.path.join(script_dir, 'data', 'obstacle', 'obstacle_toy.pt'))

    size_str = ''.join(str(n) for n in network_size)
    net_dir = os.path.join(script_dir, 'data', 'network', f'prenet_toy_{size_str}.pth')
    net.load_state_dict(torch.load(net_dir))

    # Make sure training is in CPU
    device = torch.device("cpu")
    net.to(device)

    # Constraint optimizer
    con_opt = optim.Adam(net.parameters(), lr=0.02)

    # Training loop
    iter_max = 100
    iter = 0
    a = 50
    shrinkDims = [0, 1]

    while True:
        iter += 1
        print(iter)

        # Zero the gradient
        con_opt.zero_grad()

        # Compute forward reachable set
        P_d = Z.forwardReLU(net, a)
        Q = P_d.intersect(U)

        # Check collision with MILP
        if iter % 5 == 0:
            print("Iteration: ", iter)
            isEmpty = Q.isEmpty()
            if isEmpty:
                print("Training successful!")
                break
        elif iter == iter_max:
            print("Max iteration reached!")
            break

        loss = shrinkabilityLoss(Q, shrinkDims=shrinkDims)
        loss.backward()
        con_opt.step()


if __name__ == "__main__":
    main()
