"""
Created on 6/28/21

@author: fayfayning
"""

import itertools
import numpy as np
import sys
import math
import os
import time
import multiprocessing as mp
from sys import getsizeof

from get_x import get_edges, algorithmOne, rand_assign, alg2_fetch
from tree_generation import aut, generateFreeTrees
from simulations import erd_ren, corr_erd_ren, centerAdjMatrix, calculateExpectedValueTwo

def map_fetch(map, G, edges):
    check = 1
    for edge in edges:
        # print(map)
        m1 = map[edge[0] - 1] - 1
        m2 = map[edge[1] - 1] - 1
        # print(edge, m1, m2)
        check *= G[m1][m2]
    return check

def WHM(G, H):
    n = G.shape[1]
    edges = get_edges(H)
    #print(edges)
    tree_len = len(H)
    perms = itertools.permutations(range(1, n + 1), tree_len)
    sum = 0
    for map in perms:
        sum += map_fetch(map, G, edges)
    sum = sum / aut(H)
    return sum

def fAB(A, B, K):
    T = generateFreeTrees(K)
    sum = 0
    for H in T:
        sum += aut(H) * WHM(A, H) * WHM(B, H)
    return sum

def fAB_fetch(n, p, s, K, Corr):
    # receive Y value based on algorithm 2
    # Corr is True when graphs are correlated, False when independent

    if Corr:
        C = erd_ren(n, p)
        A = corr_erd_ren(n, s, C)
        B = corr_erd_ren(n, s, C)
    else:
        A = erd_ren(n, p * s)
        B = erd_ren(n, p * s)

    A_center = centerAdjMatrix(A, n, p, s)
    B_center = centerAdjMatrix(B, n, p, s)

    #print(A_center)
    #print(B_center)
    Y_corr = fAB(A_center, B_center, K)
    return Y_corr

def run_fAB(n, p, s, K, Corr, exp_corr):
    # run one time, get Y and compare
    # Corr is True when graphs are correlated, False when independent

    #print("T:",len(T))

    Y_corr = fAB_fetch(n, p, s, K, Corr)

    print(Y_corr)
    #print('exp_Y', exp_corr)
    if Y_corr >= exp_corr:
        return [Y_corr, 1]
    else:
        return [Y_corr, 0]

def exh_sim2(m, n, p, s, K):
    # runs simulation 2
    T = generateFreeTrees(K)
    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    exp_corr = calculateExpectedValueTwo(r, n, p, s, K, len(T))
    print('expected', exp_corr)
    sum_corr = 0
    sum_ind = 0
    #corr_vals = []
    #ind_vals = []
    print('Correlated')
    for i in range(m):
        corr = run_fAB(n, p, s, K, True, exp_corr)
        sum_corr += corr[1]
        #corr_vals.append(corr[0])
    print('Independent')
    for i in range(m):
        ind = run_fAB(n, p, s, K, False, exp_corr)
        sum_ind += ind[1]
        #ind_vals.append(ind[0])
    sum_corr = sum_corr / m
    sum_ind = sum_ind / m
    ret = [sum_corr, sum_ind]
    sys.stdout.flush()
    print(ret)
    return ret

def calc_rec_Y_both(T,n,p,s,K,Corr, t):
    if Corr:
        C = erd_ren(n, p)
        A = corr_erd_ren(n, s, C)
        B = corr_erd_ren(n, s, C)
    else:
        A = erd_ren(n, p * s)
        B = erd_ren(n, p * s)

    A_center = centerAdjMatrix(A, n, p, s)
    B_center = centerAdjMatrix(B, n, p, s)

    #print(A_center)
    #print(B_center)
    Y_corr_exhaust = fAB(A_center, B_center, K)
    Y_corr_algo = alg2_fetch(T, A_center, B_center, K, t)
    return [Y_corr_algo, Y_corr_exhaust]


if __name__ == '__main__':

    args = []
    args.append(int(sys.argv[1]))
    args.append(int(sys.argv[2]))
    args.append(float(sys.argv[3]))
    args.append(float(sys.argv[4]))
    args.append(int(sys.argv[5]))

    exh_sim2(*args)



def test():
    M = [[0, 1, 1, 0, 1],
         [1, 0, 1, 1, 1],
         [1, 1, 0, 0, 0],
         [0, 1, 0, 0, 1],
         [1, 1, 0, 1, 0]]
    M = np.array(M)
    H = [0, 1, 1, 1]
    print(WHM(M, H))

    K = len(H) - 1
    print(fAB(M, M, K))

    # C = rand_assign(K, M.shape[1])
    C = [1, 2, 1, 2, 2]
    print(C)
    print(algorithmOne(H, M, C))

def timing():
    n_lst = [i * 5 for i in range(1, 10)]
    for n in n_lst:
        H = [0]
        H.extend([1 for i in range(n - 1)])
        G = erd_ren(n, 0.1)
        start = time.time()
        WHM(G, H)
        print('time', n, time.time() - start)

if __name__ == '__main__':

    #test()

    # m, n, p, s, K
    args = []
    args.append(int(sys.argv[1]))
    args.append(int(sys.argv[2]))
    args.append(float(sys.argv[3]))
    args.append(float(sys.argv[4]))
    args.append(int(sys.argv[5]))

    sys.path.append(os.getcwd())
    
    exh_sim2(*args)





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

H = [0, 1, 1, 1]
C = [1, 2, 3, 3, 4]
should be 12
"""