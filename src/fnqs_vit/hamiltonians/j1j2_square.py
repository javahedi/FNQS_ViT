# src/fnqs_vit/hamiltonians/j1j2_square.py

from fnqs_vit.hamiltonians.lattice import create_square_lattice

def heisenberg_j1j2(Lx: int, Ly: int, J2: float):
    """
    Return NN edges, NNN edges, and couplings J1=1, J2.
    """
    nn_edges, nnn_edges = create_square_lattice(Lx, Ly)
    J1 = 1.0
    return nn_edges, nnn_edges, J1, J2
