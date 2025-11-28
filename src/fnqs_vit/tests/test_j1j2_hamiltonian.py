import netket as nk
from fnqs_vit.hamiltonians.j1j2_square import heisenberg_j1j2


def test_j1j2_build():
    L = 2
    J2 = 0.5

    nn_edges, nnn_edges, J1, J2_val = heisenberg_j1j2(L, L, J2)

    assert len(nn_edges) > 0
    assert len(nnn_edges) > 0
