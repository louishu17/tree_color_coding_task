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
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            temp = random.random()
            if temp < p:
                M[i][j] = 1
                M[j][i] = 1
    return M


def calculateExpectedValueOne(m, n, p, H):
    # returns expected value based on r, n, K, p, H
    r = math.factorial(len(H)) / math.pow(len(H), len(H))
    exp_val = math.factorial(n) / math.factorial(n - len(H))
    exp_val = exp_val * math.pow(p, len(H) - 1) * r / aut(H)
    return exp_val


def simulation1(m, n, p, H):
    # runs simulation 1
    start = time.time()

    K = len(H) - 1

    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)

    t = int(math.ceil(1 / (math.pow(r, 2))))

    ev = calculateExpectedValueOne(r, n, K, p, H)

    pool = mp.Pool(mp.cpu_count())
    graphs = pool.starmap(erd_ren, [(n, p) for i in range(m)])

    resMSum = 0
    for g in graphs:
        results = pool.starmap(getX, [(H, g, K, n) for i in range(t)])

        resMSum += sum(results) / t

    pool.close()

    xH = resMSum / m
    print(xH)

    print("Ratio:", 1 - (ev / xH))
    print(time.time() - start)

    return xH

def corr_erd_ren(n, s, C):
    # creates a correlated Erdos-Renyi random graph from a random graph C
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if C[i][j] == 1:
                temp = random.random()
                if temp < s:
                    M[i][j] = 1
                    M[j][i] = 1
    return M

def centerAdjMatrix(g, n, p, s):
    # centers adjacency matrix
    for i in range(n):
        for j in range(i):
            g[i][j] = g[i][j] - p * s
            g[j][i] = g[i][j]

    return g

def calculateExpectedValueTwo(r, n, p, s, K, sizeT):
    # calculates expected value of simulation 2
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
    # runs simulation 2
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

    # args = [3, 100, 0.5, [0, 1, 1]]
    # print(simulation1(*args))

    print(sim2(100, 99, 0.5, 0.7, 3))

    print(time.time() - start)