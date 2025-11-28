# src/fnqs_vit/hamiltonians/lattice.py

import numpy as np

def create_square_lattice(Lx, Ly):
    """
    Returns:
      nn_edges      : list of (i,j)
      nnn_edges     : list of (i,j)
    """

    def idx(x, y):
        return x * Ly + y

    nn_edges = []
    nnn_edges = []

    # nearest-neighbor shifts
    nn_shifts = [(1,0), (-1,0), (0,1), (0,-1)]

    # next-nearest neighbors (diagonals)
    nnn_shifts = [(1,1), (1,-1), (-1,1), (-1,-1)]

    for x in range(Lx):
        for y in range(Ly):
            i = idx(x,y)

            # NN edges
            for dx,dy in nn_shifts:
                xn = (x + dx) % Lx
                yn = (y + dy) % Ly
                j = idx(xn, yn)
                if j > i:
                    nn_edges.append((i,j))

            # NNN edges
            for dx,dy in nnn_shifts:
                xn = (x + dx) % Lx
                yn = (y + dy) % Ly
                j = idx(xn, yn)
                if j > i:
                    nnn_edges.append((i,j))

    return nn_edges, nnn_edges
