"""
Created on 6/7/21
@author: louishu17, fayfayning, kieranlele
"""
import math
import time
import random

from automorphisms import aut
from get_x import getX, algorithm2
from tree_generation import generateFreeTrees

import multiprocessing as mp
from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def erd_ren(n, p):
    # returns adjacency matrix for randomly created Erdos-Renyi graph
    # edge included in graph with probability p
    M = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            temp = random.random()
            if temp < p:
                temp = 1
            else:
                temp = 0
            M[i][j] = temp
            M[j][i] = temp
    return M


def calculateExpectedValueOne(r, n, K, p, L):
    return r * ((math.factorial(n)) / (
                math.factorial(n - K - 1) * aut(L))) * pow(p, K)


def simulation1():
    start = time.time()

    n = 100
    p = 0.5
    m = 3

    graphs = generateErdosReyniGraphs(n, p, m)

    L = [0, 1, 1]

    K = len(L) - 1

    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)

    t = int(math.ceil(1 / (math.pow(r, 2))))

    #ev = calculateExpectedValueOne(r, n, K, p, L)

    pool = mp.Pool(mp.cpu_count())

    resMSum = 0
    for g in graphs:
        a = g.todense().tolist()
        results = pool.starmap(getX, [(L, a, K, n) for i in range(t)])

        resMSum += sum(results) / t

    pool.close()

    xH = resMSum / m
    print(xH)

    #print("Ratio:", 1 - (ev / xH))
    print(time.time() - start)

def corr_erd_ren(n, s, C):
    # creates a correlated Erdos-Renyi random graph from a random graph C
    M = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if C[i][j] == 1:
                temp = random.random()
                if temp < s:
                    temp = 1
                else:
                    temp = 0
                M[i][j] = temp
                M[j][i] = temp
    return M

def centerAdjMatrix(g, n, p, s):
    for i in range(n):
        for j in range(i):
            g[i][j] = g[i][j] - p * s
            g[j][i] = g[i][j]

    return g

def calculateExpectedValueTwo(r, n, p, s, K, sizeT):
    return 0.5 * r * r * ((math.factorial(n) * math.pow(
        (p * s * s * (1 - p)), K)) / math.factorial(n - K - 1)) * sizeT

def run_Y_comp(n, p, s, K, Corr):
    # run one time, get Y and compare
    # Corr is True when graphs are correlated, False when independent

    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    T = generateFreeTrees(K)

    if Corr:
        C = erd_ren(n, p)
        A = corr_erd_ren(n, s, C)
        B = corr_erd_ren(n, s, C)
    else:
        A = erd_ren(n, p * s)
        B = erd_ren(n, p * s)

    A_center = centerAdjMatrix(A, n, p, s)
    B_center = centerAdjMatrix(B, n, p, s)
    Y_corr = algorithm2(T, A_center, B_center, K)
    exp_corr = calculateExpectedValueTwo(r, n, p, s, K, len(T))
    print('rec_Y', Y_corr)
    print('exp_Y', exp_corr)
    if Y_corr >= exp_corr:
        return 1
    else:
        return 0

def sim2(m, n, p, s, K):
    if K > (math.log(n) / math.log(math.log(n))):
        print('K too large')
        return
    if s * s <= 0.33833:
        print('s too small')
        return
    sum_corr = 0
    for i in range(m):
        sum_corr += run_Y_comp(n, p, s, K, True)
    sum_ind = 0
    for i in range(m):
        sum_ind += run_Y_comp(n, p, s, K, False)
    sum_corr = sum_corr / m
    sum_ind = sum_ind / m
    return [sum_corr, sum_ind]

if __name__ == '__main__':

    start = time.time()
    print(sim2(20, 99, 0.5, 0.7, 3))