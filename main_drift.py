import numpy as np
import os
import torch
import torch.optim as optim
from util.MILPLoss import shrinkabilityLoss
from util.hybZono import HybZono, extractReLUNetParams, ReLUGraph, selectionMatrix
from util.net import Net


def main():
    """
    Main script for the drift-parking example. See Section VI.B of chung2025provably.

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

    # Setup network
    f_network_size = (3, 6, 6, 6, 6, 2)
    fnet = Net(f_network_size)
    u_network_size = (2, 20, 2)
    unet = Net(u_network_size)

    # Read in the files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    X_0 = HybZono.load(os.path.join(script_dir, 'data', 'input', 'input_drift.pt'))
    U = HybZono.load(os.path.join(script_dir, 'data', 'obstacle', 'obstacle_drift.pt'))
    unet.load_state_dict(torch.load(os.path.join(script_dir, 'data', 'network', f'prenet_ctrl_drift_2202.pth')))
    fnet.load_state_dict(torch.load(os.path.join(script_dir, 'data', 'network', f'net_dyn_drift_366662.pth')))

    # Lock the parameters in the dynamics matrix
    for param in fnet.parameters():
        param.requires_grad = False

    # Set up normalization
    V_hi = 11.
    V_lo = 9.
    beta_hi = (2. / 9.) * np.pi
    beta_lo = np.pi / 6.
    normalize_factor = (V_hi - V_lo) / (beta_hi - beta_lo)

    # Set up the set of time
    c_T = torch.tensor([3.9])
    Gc_T = 3.9 * torch.eye(1)
    T = HybZono(Gc_T, None, c_T, None, None, None)

    # Make sure training is in CPU
    device = torch.device("cpu")
    unet.to(device)

    # Constraint optimizer
    con_opt = optim.Adam(unet.parameters(), lr=0.005)

    # Training loop
    iter_max = 100
    iter = 0
    shrinkDims = [0, 1]  # n_r in chung2025provably
    a = 40

    while True:
        iter += 1

        # Zero the gradient
        con_opt.zero_grad()

        # Compute forward reachable tube
        P_g = forwardDrift(X_0, T, fnet, unet, normalize_factor, a)
        Q = P_g.intersect(U)

        # Check collision with MILP
        if iter % 5 == 0:
            print("Iteration: ", iter)
            isEmpty = Q.isEmpty()
            if isEmpty:
                print("Training successful!")
                break
        if iter == iter_max:
            print("Max iteration reached!")
            break

        loss = shrinkabilityLoss(Q, shrinkDims=shrinkDims)
        loss.backward()
        con_opt.step()


def forwardDrift(HZ_in, T, fnet, unet, normalize_factor=1., a=1000):
    """
    Compute the forward reachable tube of the drift-parking system.

    References:
    1. Chung, Long Kiu, and Shreyas Kousik. "Provably-Safe Neural Network Training Using Hybrid Zonotope Reachability
       Analysis." arXiv preprint arXiv:2501.13023 (2025).
    2. Ortiz, Joshua, et al. "Hybrid zonotopes exactly represent ReLU neural networks." 2023 62nd IEEE Conference on
       Decision and Control (CDC). IEEE, 2023.

    INPUTS:
    HZ_in: HybZono; input hybrid zonotope
    T: HybZono; set of time to be analyzed
    fnet: Net; dynamics network for drifting
    unet: Net; controller network for drifting
    normalize_factor: float; normalization factor for beta; defaults to 1
    a: float; radius of the domain of the ReLU functions; see ortiz2023hybrid; defaults to 1000

    OUTPUTS:
    HZ_out: HybZono; forward reachable tube of HZ_in

    Author: Long Kiu Chung
    Created: 2025/03/25
    Updated: 2025/05/01
    """

    # Forward reachability through controller
    HZ_XK = HZ_in.forwardReLUGraph(unet, a)   # x_0, k

    # Unnormalize beta
    HZ_XK = HZ_XK.affineMap(torch.diag(torch.tensor([1., 1., 1, 1. / normalize_factor])))

    # Add time dimension to the reachable set
    HZ_XKT = HZ_XK.cartProd(T)  # x_0, k, t

    # Extract variables
    n = HZ_XKT.n

    # Extract weights and biases from the neural network
    Ws, ws = extractReLUNetParams(fnet)

    # Get depth of the neural network
    l = len(Ws)

    # Neural network hybrid zonotope
    NN = HZ_XKT

    v_indices = []
    x_indices = []
    n_in = HZ_in.n
    x_indices.append(list(range(n_in, n)))
    idx_max = n

    for i in range(l - 1):
        n2, n1 = Ws[i].shape
        relu_layer = ReLUGraph(n2, a)

        # Keep track of indices for adding layer-to-layer connections
        v_indices.append(list(range(idx_max, idx_max + n2)))
        idx_max += n2
        x_indices.append(list(range(idx_max, idx_max + n2)))
        idx_max += n2

        # Add layer to hybrid zonotope neural network
        NN = NN.cartProd(relu_layer)

    # Make layer-to-layer connections
    for i in range(l - 1):
        W = Ws[i]
        w = ws[i]
        Rx = selectionMatrix(x_indices[i], NN.n)
        Rv = selectionMatrix(v_indices[i], NN.n)
        NN = NN.hyperplaneIntersect(Rv - W @ Rx, w)

    # Make connections from final layer to output
    Rx = selectionMatrix(list(range(n)), NN.n)
    RxL = selectionMatrix(x_indices[-1], NN.n)

    X = NN.affineMap(Rx)
    Y = NN.affineMap(Ws[-1] @ RxL, ws[-1])

    # Assemble output hybrid zonotope
    Gc_out = torch.cat((X.Gc, Y.Gc))
    Gb_out = torch.cat((X.Gb, Y.Gb))
    c_out = torch.cat((X.c, Y.c))
    Ac_out = X.Ac
    Ab_out = X.Ab
    b_out = X.b

    HZ_out = HybZono(Gc_out, Gb_out, c_out, Ac_out, Ab_out, b_out)   # x_0, k, t, x_t

    # Select t, x_t
    R = torch.zeros((n_in + 1, HZ_out.n))
    R[0, -(n_in + 1)] = 1.
    R[-n_in:, :n_in] = torch.eye(n_in)
    R[-n_in:, -n_in:] = torch.eye(n_in)

    HZ_out = HZ_out.affineMap(R)

    return HZ_out


if __name__ == "__main__":
    main()
