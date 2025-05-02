import torch
import gurobipy as gp
from gurobipy import GRB
from scipy.io import savemat


class HybZono(object):
    """
    Hybrid zonotope class for PyTorch. A hybrid zonotope is parameterized by Gc, Gb, c, Ac, Ab, b as {Gc*zc + Gb*zb + c
    | Ac*zc + Ab*zb = b, ||zc||_inf <= 1, zb in {-1, 1}^nGb}

    This code is adapted from the following codebases:
    1. Chung, Long Kiu, et al. "Constrained feedforward neural network training via reachability analysis." arXiv
       preprint arXiv:2107.07696 (2021).
    2. Koeln, Justin, et al. "zonoLAB: A MATLAB toolbox for set-based control systems analysis using hybrid zonotopes."
       2024 American Control Conference (ACC). IEEE, 2024.

    TODO:
        1. Implement union, set difference, Minkowski sum, and various other hybrid zonotope operations
        2. Implement order reduction
        3. Implement plotting
        For now, use zonoLAB for these functions

    Author: Long Kiu Chung
    Created: 2025/01/02
    Updated: 2025/04/29
    """

    ### Constructors
    def __init__(self, Gc=None, Gb=None, c=None, Ac=None, Ab=None, b=None):
        """
        Constructor for a hybrid zonotope.

        INPUTS:
        Gc: torch tensor, size n*nGc; continuous generator matrix; defaults as an empty array
        Gb: torch tensor, size n*nGb; binary generator matrix; defaults as an empty array
        c: torch tensor, size n; center; defaults as a zero vector
        Ac: torch tensor, size nC*nGc; continuous constraint matrix; defaults as an empty array
        Ab: torch tensor, size nC*nGb; binary constraint matrix; defaults as an empty array
        b: torch tensor, size nC; constraint vector; defaults as an empty array

        OUTPUTS:
        N/A

        Author: Long Kiu Chung
        Created: 2025/01/02
        Updated: 2025/04/29
        """

        if (Gc is None or Gc.numel() == 0) and (Gb is None or Gb.numel() == 0):
            raise Exception("Please specify either Gc or Gb.")

        if Gc is None or Gc.numel() == 0:  # Only Gb is specified
            self.Gb = Gb
            self.n, self.nGb = Gb.shape
            self.Gc = torch.zeros((self.n, 0))
            self.nGc = 0
        elif Gb is None or Gb.numel() == 0:  # Only Gc is specified
            self.Gc = Gc
            self.n, self.nGc = Gc.shape
            self.Gb = torch.zeros((self.n, 0))
            self.nGb = 0
        else:  # Both Gc and Gb are specified
            self.Gc = Gc
            self.Gb = Gb
            self.n, self.nGc = Gc.shape
            self.nGb = Gb.size(1)

        if c is None:
            self.c = torch.zeros(self.n)
        else:
            self.c = c

        if b is None or b.numel() == 0:  # Neither Ac nor Ab are specified
            self.Ac = torch.zeros((0, self.nGc))
            self.Ab = torch.zeros((0, self.nGb))
            self.b = torch.zeros(0)
            self.nC = 0
        else:
            self.b = b
            self.nC = b.size(0)

            if (Ac is None or Ac.numel() == 0) and (Ab is None or Ab.numel() == 0):
                raise Exception("Please specify either Ac or Ab if you specify b.")

            if Ac is None or Ac.numel() == 0: # Only Ab is specified
                self.Ab = Ab
                self.Ac = torch.zeros((self.nC, self.nGc))
            elif Ab is None or Ab.numel() == 0: # Only Ac is specified
                self.Ac = Ac
                self.Ab = torch.zeros((self.nC, self.nGb))
            else:  # Both Ac and Ab are specified
                self.Ac = Ac
                self.Ab = Ab

    ### Operations
    def intersect(self, other):
        """
        Compute the intersection between two hybrid zonotopes. i.e. {z in self | z in other}.

        This method is based on:
        1. Bird, Trevor J., et al. "Hybrid zonotopes: A new set representation for reachability analysis of mixed
           logical dynamical systems." Automatica 154 (2023): 111107.

        INPUTS:
        self: HybZono; first hybrid zonotope to be intersected
        other: HybZono; second hybrid zonotope to be intersected

        OUTPUTS:
        HZ_out: HybZono; intersection of self and other

        Author: Long Kiu Chung
        Created: 2025/01/02
        Updated: 2025/04/29
        """

        # Extract variables
        Gc_z = self.Gc
        Gb_z = self.Gb
        c_z = self.c
        Ac_z = self.Ac
        Ab_z = self.Ab
        b_z = self.b
        n = self.n

        Gc_y = other.Gc
        Gb_y = other.Gb
        c_y = other.c
        Ac_y = other.Ac
        Ab_y = other.Ab
        b_y = other.b
        nGc_y = other.nGc
        nGb_y = other.nGb

        # Proposition 7 in bird2023hybrid
        Gc_out = torch.cat((Gc_z, torch.zeros((n, nGc_y))), 1)
        Gb_out = torch.cat((Gb_z, torch.zeros((n, nGb_y))), 1)
        c_out = c_z
        Ac_out = torch.block_diag(Ac_z, Ac_y)
        Ac_out = torch.cat((Ac_out, torch.cat((Gc_z, -Gc_y), 1)))
        Ab_out = torch.block_diag(Ab_z, Ab_y)
        Ab_out = torch.cat((Ab_out, torch.cat((Gb_z, -Gb_y), 1)))
        b_out = torch.cat((b_z, b_y, c_y - c_z))

        HZ_out = HybZono(Gc_out, Gb_out, c_out, Ac_out, Ab_out, b_out)

        return HZ_out

    def generalIntersect(self, other, R):
        """
        Compute the generalized intersection between two hybrid zonotopes. i.e. {z in self | R*z in other}.

        This method is based on:
        1. Bird, Trevor J., et al. "Hybrid zonotopes: A new set representation for reachability analysis of mixed
           logical dynamical systems." Automatica 154 (2023): 111107.

        INPUTS:
        self: HybZono; first hybrid zonotope to be intersected
        other: HybZono; second hybrid zonotope to be intersected
        R: torch tensor, size n_y*n_z; intersection matrix

        OUTPUTS:
        HZ_out: HybZono; generalized intersection of self and other under R

        Author: Long Kiu Chung
        Created: 2025/01/02
        Updated: 2025/04/29
        """

        # Extract variables
        Gc_z = self.Gc
        Gb_z = self.Gb
        c_z = self.c
        Ac_z = self.Ac
        Ab_z = self.Ab
        b_z = self.b
        n_z = self.n

        Gc_y = other.Gc
        Gb_y = other.Gb
        c_y = other.c
        Ac_y = other.Ac
        Ab_y = other.Ab
        b_y = other.b
        nGc_y = other.nGc
        nGb_y = other.nGb

        # Proposition 7 in bird2023hybrid
        Gc_out = torch.cat((Gc_z, torch.zeros((n_z, nGc_y))), 1)
        Gb_out = torch.cat((Gb_z, torch.zeros((n_z, nGb_y))), 1)
        c_out = c_z
        Ac_out = torch.block_diag(Ac_z, Ac_y)
        Ac_out = torch.cat((Ac_out, torch.cat((R @ Gc_z, -Gc_y), 1)))
        Ab_out = torch.block_diag(Ab_z, Ab_y)
        Ab_out = torch.cat((Ab_out, torch.cat((R @ Gb_z, -Gb_y), 1)))
        b_out = torch.cat((b_z, b_y, c_y - R @ c_z))

        HZ_out = HybZono(Gc_out, Gb_out, c_out, Ac_out, Ab_out, b_out)

        return HZ_out

    def hyperplaneIntersect(self, H, h):
        """
        Compute the intersection between a hybrid zonotope and hyperplanes, defined as {z in self | H*z = h}.

        INPUTS:
        self: HybZono; hybrid zonotope to be intersected
        H: torch tensor, size nCh*n; constraint matrix of the hyperplanes
        h: torch tensor, size nCh; constraint vector of the hyperplanes

        OUTPUTS:
        HZ_out: HybZono; intersection of self and hyperplanes defined by H and h

        Author: Long Kiu Chung
        Created: 2025/04/29
        Updated: 2025/04/29
        """

        # Extract variables
        Gc = self.Gc
        Gb = self.Gb
        c = self.c
        Ac = self.Ac
        Ab = self.Ab
        b = self.b

        # Compute output
        Ac_out = torch.cat((Ac, H @ Gc))
        Ab_out = torch.cat((Ab, H @ Gb))
        b_out = torch.cat((b, h - H @ c))

        HZ_out = HybZono(Gc, Gb, c, Ac_out, Ab_out, b_out)

        return HZ_out

    def affineMap(self, R=None, r=None):
        """
        Compute the affine map of a hybrid zonotope. i.e. {R*z + r | z in self}.

        This method is based on:
        1. Bird, Trevor J., et al. "Hybrid zonotopes: A new set representation for reachability analysis of mixed
           logical dynamical systems." Automatica 154 (2023): 111107.
        2. Koeln, Justin, et al. "zonoLAB: A MATLAB toolbox for set-based control systems analysis using hybrid
           zonotopes." 2024 American Control Conference (ACC). IEEE, 2024.

        INPUTS:
        self: HybZono; hybrid zonotope to be transformed
        R: torch tensor, size n_out*n; transformation matrix; defaults as identity matrix
        r: torch tensor, size n_out; transformation vector; defaults as zero vector

        OUTPUTS:
        HZ_out: TorchHybZono; affine map of self under R and r

        Author: Long Kiu Chung
        Created: 2025/01/02
        Updated: 2025/04/29
        """

        if R is None and r is None:  # Neither R nor r are specified
            return self

        if r is None:  # Only R is specified
            n_out = R.size(0)
            r = torch.zeros(n_out)
        elif R is None:  # Only r is specified
            n_out = r.size(0)
            R = torch.eye(n_out)

        # Apply affine map
        HZ_out = HybZono(R @ self.Gc, R @ self.Gb, R @ self.c + r, self.Ac, self.Ab, self.b)

        return HZ_out

    def cartProd(self, other):
        """
        Compute the Cartesian product of two hybrid zonotopes. i.e. {[z^T, y^T]^T | z in self, y in other}.

        This method is based on:
        1. Koeln, Justin, et al. "zonoLAB: A MATLAB toolbox for set-based control systems analysis using hybrid
           zonotopes." 2024 American Control Conference (ACC). IEEE, 2024.

        INPUTS:
        self: HybZono; first hybrid zonotope input of the Cartesian product
        other: HybZono; second hybrid zonotope input of the Cartesian product

        OUTPUTS:
        HZ_out: HybZono; Cartesian product of self and other

        Author: Long Kiu Chung
        Created: 2025/01/02
        Updated: 2025/04/29
        """

        # From zonoLAB
        Gc_out = torch.block_diag(self.Gc, other.Gc)
        Gb_out = torch.block_diag(self.Gb, other.Gb)
        c_out = torch.cat((self.c, other.c))
        Ac_out = torch.block_diag(self.Ac, other.Ac)
        Ab_out = torch.block_diag(self.Ab, other.Ab)
        b_out = torch.cat((self.b, other.b))

        HZ_out = HybZono(Gc_out, Gb_out, c_out, Ac_out, Ab_out, b_out)

        return HZ_out

    def power(self, m):
        """
        Compute the Cartesian product of a hybrid zonotope with itself m times.

        INPUTS:
        self: HybZono; input hybrid zonotope
        m: int; number of times to apply the Cartesian product

        OUTPUTS:
        HZ_out: HybZono; self after applying Cartesian product to itself m times

        Author: Long Kiu Chung
        Created: 2025/01/02
        Updated: 2025/04/29
        """

        # Concatenate repeatedly
        c_out = self.c.repeat(m)
        b_out = self.b.repeat(m)

        # Apply block diag repeatedly
        Gcs = [self.Gc]*m
        Gbs = [self.Gb]*m
        Acs = [self.Ac]*m
        Abs = [self.Ab]*m

        Gc_out = torch.block_diag(*Gcs)
        Gb_out = torch.block_diag(*Gbs)
        Ac_out = torch.block_diag(*Acs)
        Ab_out = torch.block_diag(*Abs)

        HZ_out = HybZono(Gc_out, Gb_out, c_out, Ac_out, Ab_out, b_out)

        return HZ_out

    def isEmpty(self, isMute=True):
        """
        Check if a hybrid zonotope is empty.

        INPUTS:
        self: HybZono; hybrid zonotope to check for emptiness
        isMute: bool; True if suppressing output messages

        OUTPUTS:
        isEmpty: bool; True if self is empty, False otherwise

        Author: Long Kiu Chung
        Created: 2025/01/15
        Updated: 2025/04/29
        """

        # Extract variables
        Ac = self.Ac
        Ab = self.Ab
        b = self.b
        nGc = self.nGc
        nGb = self.nGb
        nC = self.nC

        # Variable type for Gurobi
        vType = [GRB.CONTINUOUS]*nGc
        vType += [GRB.BINARY]*nGb

        env = gp.Env(empty=True)
        if isMute:
            env.setParam("OutputFlag", 0)  # Suppresses most messages
            env.setParam("LogToConsole", 0)  # Suppresses all console output, including warnings
        env.start()

        # Set up model
        model = gp.Model(env=env)
        nG = nGc + nGb
        x = model.addVars(nG, vtype=vType, lb=[0.]*nG, ub=[1.]*nG)

        # Set equality constraints
        A_gp = torch.cat((2*Ac, 2*Ab), 1)
        b_gp = b + Ac @ torch.ones(nGc) + Ab @ torch.ones(nGb)
        A_list = A_gp.tolist()
        b_list = b_gp.tolist()
        model.addConstrs(x.prod(A_list[i]) == b_list[i] for i in range(nC))

        # Set objective
        model.setObjective(0, GRB.MINIMIZE)

        # Optimize model
        model.optimize()

        return not model.status == GRB.OPTIMAL

    def projection(self, dims):
        """
        Project a hybrid zonotope into a lower dimension.

        This method is based on:
        1. Koeln, Justin, et al. "zonoLAB: A MATLAB toolbox for set-based control systems analysis using hybrid
           zonotopes." 2024 American Control Conference (ACC). IEEE, 2024.

        INPUTS:
        self: HybZono; hybrid zonotope to be projected
        dims: list of int; dimensions to project onto (0-indexing)

        OUTPUTS:
        HZ_out: HybZono; projected hybrid zonotope

        Author: Long Kiu Chung
        Created: 2025/02/03
        Updated: 2025/04/29
        """

        # Projection
        c_out = self.c[dims]
        Gc_out = self.Gc[dims, :]
        Gb_out = self.Gb[dims, :]

        # Output
        HZ_out = HybZono(Gc_out, Gb_out, c_out, self.Ac, self.Ab, self.b)

        return HZ_out

    def overapprox(self):
        """
        Over-approximate a hybrid zonotope as a single constrained zonotope (expressed as a hybrid zonotope still).

        See robbins2024mixed and zhang2023reachability on scenarios where this algorithm is equal to the convex hull of
        the hybrid zonotope.

        This method is based on:
        1. Robbins, Joshua A., et al. "Mixed-Integer MPC-Based Motion Planning Using Hybrid Zonotopes with Tight
           Relaxations." arXiv preprint arXiv:2411.01286 (2024).
        2. Zhang, Yuhao, and Xiangru Xu. "Reachability analysis and safety verification of neural feedback systems via
           hybrid zonotopes." 2023 American Control Conference (ACC). IEEE, 2023.

        INPUTS:
        self: HybZono; hybrid zonotope to be over-approximated

        OUTPUTS:
        HZ_out: HybZono; over-approximation of self

        Author: Long Kiu Chung
        Created: 2025/03/11
        Updated: 2025/04/29
        """

        # Convert binary constraints to continuous constraints
        Ac_out = torch.hstack((self.Ac, self.Ab))
        Gc_out = torch.hstack((self.Gc, self.Gb))

        # Output
        HZ_out = HybZono(Gc_out, None, self.c, Ac_out, None, self.b)

        return HZ_out

    def ReLU(self, a=1000):
        """
        Apply the ReLU function to a hybrid zonotope.

        This method is based on:
        1. Ortiz, Joshua, et al. "Hybrid zonotopes exactly represent ReLU neural networks." 2023 62nd IEEE Conference on
           Decision and Control (CDC). IEEE, 2023.
        2. Zhang, Yuhao, Hang Zhang, and Xiangru Xu. "Backward reachability analysis of neural feedback systems using
           hybrid zonotopes." IEEE Control Systems Letters 7 (2023): 2779-2784.

        INPUTS:
        self: HybZono; input hybrid zonotope
        a: float; radius of the domain of the ReLU functions; see ortiz2023hybrid; defaults to 1000

        OUTPUTS:
        HZ_out: HybZono; output hybrid zonotope

        Author: Long Kiu Chung
        Created: 2025/02/28
        Updated: 2025/04/29
        """

        # Extract parameters
        n = self.n

        # Apply ReLU
        HZ_ReLU = ReLUGraph(n, a)
        G = HZ_ReLU.generalIntersect(self, torch.cat((torch.eye(n), torch.zeros((n, n))), 1))
        HZ_out = G.affineMap(torch.cat((torch.zeros((n, n)), torch.eye(n)), 1))

        return HZ_out

    ### Neural Network Reachability
    def forwardReLU(self, net, a=1000):
        """
        Compute the forward reachable set of a hybrid zonotope through a ReLU neural network.

        This method is based on:
        1. Ortiz, Joshua, et al. "Hybrid zonotopes exactly represent ReLU neural networks." 2023 62nd IEEE Conference on
           Decision and Control (CDC). IEEE, 2023.
        2. Zhang, Yuhao, Hang Zhang, and Xiangru Xu. "Backward reachability analysis of neural feedback systems using
           hybrid zonotopes." IEEE Control Systems Letters 7 (2023): 2779-2784.

        INPUTS:
        self: HybZono; input hybrid zonotope
        net: Net with fully connected layers and ReLU activation between every layer; see PyTorch documentation
        a: float; radius of the domain of the ReLU functions; see ortiz2023hybrid; defaults to 1000

        OUTPUTS:
        HZ_out: HybZono; forward reachable set of self through net

        Author: Long Kiu Chung
        Created: 2025/01/03
        Updated: 2025/04/29
        """

        n_out = net.layers[-1].out_features
        HZ_graph = self.forwardReLUGraph(net, a)
        n_graph = HZ_graph.n
        R = torch.cat((torch.zeros((n_out, n_graph - n_out)), torch.eye(n_out)), 1)
        HZ_out = HZ_graph.affineMap(R)

        return HZ_out

    def forwardReLUGraph(self, net, a=1000):
        """
        Compute the graph of a hybrid zonotope through a ReLU neural network. i.e. {[x; net(x)] | x in self}

        This method is based on:
        1. Koeln, Justin, et al. "zonoLAB: A MATLAB toolbox for set-based control systems analysis using hybrid
           zonotopes." 2024 American Control Conference (ACC). IEEE, 2024.
        2. Ortiz, Joshua, et al. "Hybrid zonotopes exactly represent ReLU neural networks." 2023 62nd IEEE Conference on
           Decision and Control (CDC). IEEE, 2023.

        INPUTS:
        self: HybZono; input hybrid zonotope
        net: Net with fully connected layers and ReLU activation between every layer; see PyTorch documentation
        a: float; radius of the domain of the ReLU functions; see ortiz2023hybrid; defaults to 1000

        OUTPUTS:
        HZ_out: HybZono; graph of self through net

        Author: Long Kiu Chung
        Created: 2025/01/28
        Updated: 2025/04/30
        """

        # Extract variables
        n = self.n

        # Extract weights and biases from the neural network
        Ws, ws = extractReLUNetParams(net)

        # Get depth of the neural network
        l = len(Ws)

        # Neural network hybrid zonotope
        NN = self

        v_indices = []
        x_indices = []
        x_indices.append(list(range(n)))
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
        Rx = selectionMatrix(x_indices[0], NN.n)
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

        HZ_out = HybZono(Gc_out, Gb_out, c_out, Ac_out, Ab_out, b_out)

        return HZ_out

    ### Saving and Loading
    def save(self, filename='hybZono.pt'):
        """
        Save the hybrid zonotope as a .pt file.

        INPUTS:
        self: HybZono; hybrid zonotope to be saved
        filename: file-like object or str or os.PathLike object with a file name; see documentation for torch.save;
                  defaults to 'hybZono.pt'

        OUTPUTS:
        N/A

        Author: Long Kiu Chung
        Created: 2025/04/30
        Updated: 2025/04/30
        """

        torch.save({'Gc': self.Gc, 'Gb': self.Gb, 'c': self.c, 'Ac': self.Ac, 'Ab': self.Ab, 'b': self.b}, filename)

    def saveToMATLAB(self, filename='hybZono.mat'):
        """
        Save the hybrid zonotope to a .mat file.

        INPUTS:
        self: TorchHybZono; hybrid zonotope to be saved
        filename: file-like object or str; see documentation for scipy.io.savemat; defaults to 'hybZono.mat'

        OUTPUTS:
        N/A

        Author: Long Kiu Chung
        Created: 2025/01/15
        Updated: 2025/05/01
        """

        # Move the object to NumPy for saving
        Gc = self.Gc.detach().double().numpy()
        Gb = self.Gb.detach().double().numpy()
        c = self.c.detach().double().numpy().reshape(-1, 1)
        Ac = self.Ac.detach().double().numpy()
        Ab = self.Ab.detach().double().numpy()
        b = self.b.detach().double().numpy().reshape(-1, 1)

        # Save the tensors
        savemat(filename, {'Gc': Gc, 'Gb': Gb, 'c': c, 'Ac': Ac, 'Ab': Ab, 'b': b})

    @classmethod
    def load(cls, filename):
        """
        Load a hybrid zonotope from a .pt file.

        INPUTS:
        cls: HybZono class
        filename: file-like object or str or os.PathLike object with a file name; see documentation for torch.load

        OUTPUTS:
        HZ_out: HybZono; loaded hybrid zonotope

        Author: Long Kiu Chung
        Created: 2025/04/30
        Updated: 2025/04/30
        """

        # Load the parameters
        data = torch.load(filename)
        Gc = data.get('Gc', None)
        Gb = data.get('Gb', None)
        c = data.get('c', None)
        Ac = data.get('Ac', None)
        Ab = data.get('Ab', None)
        b = data.get('b', None)

        return cls(Gc, Gb, c, Ac, Ab, b)


# Helper Functions
def ReLUGraph(nK, a=1000):
    """
    Express the graph of a ReLU function as a hybrid zonotope.

    This method is based on:
    1. Ortiz, Joshua, et al. "Hybrid zonotopes exactly represent ReLU neural networks." 2023 62nd IEEE Conference on
       Decision and Control (CDC). IEEE, 2023.
    2. Zhang, Yuhao, Hang Zhang, and Xiangru Xu. "Backward reachability analysis of neural feedback systems using hybrid
       zonotopes." IEEE Control Systems Letters 7 (2023): 2779-2784.

    INPUTS:
    nK: int; input dimension of the ReLU function
    a: float; radius of the domain of the ReLU function; see ortiz2023hybrid; defaults to 1000

    OUTPUTS:
    HZ_ReLU: HybZono; graph of a ReLU function

    Author: Long Kiu Chung
    Created: 2025/01/03
    Updated: 2025/04/29
    """

    # Combine equation 9 of ortiz2023hybrid with equation 3 of zhang2023backward
    Gc_ReLU = torch.kron(torch.eye(nK), torch.tensor([-a/2., -a/2., 0., 0.]))
    Gc_ReLU = torch.cat((Gc_ReLU, torch.kron(torch.eye(nK), torch.tensor([0., -a/2., 0., 0.]))))
    Gb_ReLU = (-a/2.)*torch.eye(nK)
    Gb_ReLU = torch.cat((Gb_ReLU, torch.zeros((nK, nK))))
    c_ReLU = (a/2.)*torch.ones(2*nK)
    Ac_ReLU = torch.kron(torch.eye(nK), torch.cat((torch.eye(2), torch.eye(2)), 1))
    Ab_ReLU = torch.kron(torch.eye(nK), torch.tensor([[1.], [-1.]]))
    b_ReLU = torch.ones(2*nK)

    HZ_ReLU = HybZono(Gc_ReLU, Gb_ReLU, c_ReLU, Ac_ReLU, Ab_ReLU, b_ReLU)

    return HZ_ReLU


def extractReLUNetParams(net):
    """
    Given a ReLU network, return its weights and biases as lists.

    This method is based on:
    1. Chung, Long Kiu, et al. "Constrained feedforward neural network training via reachability analysis." arXiv
       preprint arXiv:2107.07696 (2021).

    INPUTS:
    net: Net with fully connected layers and ReLU activation between every layer; see PyTorch documentation

    OUTPUTS:
    Ws: list of torch tensor; weights of net
    ws: list of torch tensor; biases of net

    Author: Long Kiu Chung
    Created: 2025/01/28
    Updated: 2025/01/28
    """

    # Extract weights and biases from the neural network
    Ws = []
    ws = []

    idx = 0
    for param in net.parameters():
        if idx % 2 == 0:  # "even" parameters are weights
            Ws.append(param)
        else:  # "odd" parameters are biases
            ws.append(param)
        idx += 1

    return Ws, ws


def selectionMatrix(dims, n_in):
    """
    Build a projection matrix to project from n_in to indexed dimensions.

    This method is based on:
    1. Koeln, Justin, et al. "zonoLAB: A MATLAB toolbox for set-based control systems analysis using hybrid zonotopes."
       2024 American Control Conference (ACC). IEEE, 2024.

    INPUTS:
    dims: list of int; dimensions to project onto
    n_in: int; original dimension

    OUTPUTS:
    R: torch tensor, size n_out*n_in; projection matrix

    Author: Long Kiu Chung
    Created: 2025/02/04
    Updated: 2025/02/04
    """

    # From reluNN.m in zonoLAB
    n_out = len(dims)
    R = torch.zeros((n_out, n_in))
    R[:, dims] = torch.eye(n_out)

    return R
