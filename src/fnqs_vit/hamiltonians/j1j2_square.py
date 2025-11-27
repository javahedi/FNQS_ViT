import netket as nk


def make_square_lattice(L: int):
    """
    Build a square lattice with PBC and next-nearest neighbors included.
    NetKet's Square graph automatically generates NN (+ diagonals) up to
    max_neighbor_order.

    Parameters
    ----------
    L : int
        Linear lattice size (Lx = Ly = L)

    Returns
    -------
    nk.graph.Square
    """
    return nk.graph.Square(L, max_neighbor_order=2)  # includes NN and NNN


def heisenberg_j1j2(L: int, J2: float):
    """
    Creates the J1-J2 Heisenberg Hamiltonian using NetKet's built-in
    Heisenberg operator with J=[J1, J2].

    Parameters
    ----------
    L : int
        Linear size of the lattice
    J2 : float
        Next-nearest neighbor coupling (γ)

    Returns
    -------
    H : nk.operator.LocalOperator
    hilbert : nk.hilbert.Hilbert
    graph : nk.graph.Graph
    """
    graph = make_square_lattice(L)
    hilbert = nk.hilbert.Spin(s=1/2, N=graph.n_nodes)

    J1 = 1.0

    H = nk.operator.Heisenberg(
        hilbert=hilbert,
        graph=graph,
        J=[J1, J2]   # NN and NNN coupling
    )
    return H, hilbert, graph
