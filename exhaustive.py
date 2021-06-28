"""
Created on 6/28/21

@author: fayfayning
"""

import itertools
import numpy as np

from get_x import get_edges
from tree_generation import aut, generateFreeTrees

def WHM(G, H):
    n = G.shape[1]
    edges = get_edges(H)
    #print(edges)
    tree_len = len(H)
    combs = list(itertools.combinations(range(1, n + 1), tree_len))
    sum = 0
    for subset in combs:
        perm = list(itertools.permutations(subset, tree_len))
        for map in perm:
            check = 1
            for edge in edges:
                #print(map)
                m1 = map[edge[0] - 1] - 1
                m2 = map[edge[1] - 1] - 1
                #print(edge, m1, m2)
                if G[m1][m2] == 0:
                    check = 0
                    break
            sum += check
    fin = sum / aut(H)
    return fin

def fAB(A, B, K):
    T = generateFreeTrees(K)
    sum = 0
    for H in T:
        sum += aut(H) * WHM(A, H) * WHM(B, H)
    return sum

if __name__ == '__main__':
    M = [[0, 1, 1, 0, 1],
         [1, 0, 1, 1, 1],
         [1, 1, 0, 0, 0],
         [0, 1, 0, 0, 1],
         [1, 1, 0, 1, 0]]
    M = np.array(M)
    H = [0, 1, 1, 1]
    print(WHM(M, H))

    K = 3
    print(fAB(M, M, K))


"""
M = [[0, 1, 1, 0, 1],
         [1, 0, 1, 1, 1],
         [1, 1, 0, 0, 0],
         [0, 1, 0, 0, 1],
         [1, 1, 0, 1, 0]]
H = [0, 1, 1]
WHM = 14

H = [0, 1, 1, 1]
WMH = 6
"""