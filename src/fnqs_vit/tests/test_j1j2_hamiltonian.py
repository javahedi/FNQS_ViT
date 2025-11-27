import netket as nk
from fnqs_vit.hamiltonians.j1j2_square import heisenberg_j1j2


def test_j1j2_build():
    L = 2
    J2 = 0.5

    H, hilbert, graph = heisenberg_j1j2(L, J2)

    assert isinstance(H, nk.operator.LocalOperator)
    assert hilbert.size == L * L
    assert len(graph.edges()) > 0  # lattice constructed
