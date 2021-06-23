"""
Created on 6/7/21
@author: louishu17, fayfayning, kieranlele
"""
import math
import time
import random
from decimal import Decimal

from automorphisms import aut
from get_x import algorithmOne, algorithm2, rand_assign
from tree_generation import generateFreeTrees, center_tree

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
        for j in range(i):
            temp = random.random()
            if temp < p:
                M[i][j] = 1
                M[j][i] = 1
    return M


def calculateExpectedValueOne(m, n, p, H):
    # returns expected value based on r, n, K, p, H
    print(m, n, p, H)
    r = Decimal(math.factorial(len(H)) / math.pow(len(H), len(H)))
    exp_val = Decimal(r * math.factorial(n) / math.factorial(n - len(H)))
    exp_val *= Decimal(math.pow(p, len(H) - 1) / aut(H))
    return float(exp_val)


def simulation1(m, n, p, H):
    # runs simulation 1
    start = time.time()

    K = len(H) - 1

    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)

    t = int(math.ceil(1 / (math.pow(r, 2))))

    ev = calculateExpectedValueOne(m, n, p, H)

    pool = mp.Pool(mp.cpu_count())
    graphs = pool.starmap(erd_ren, [(n, p) for i in range(m)])

    resMSum = 0
    for g in graphs:
        color_nodes = pool.starmap(rand_assign, [(K, n) for i in range(t)])
        results = pool.starmap(algorithmOne, [(H, g, C) for C in color_nodes])

        resMSum += sum(results) / t

    pool.close()

    xH = resMSum / m
    print("xH:", xH)
    print("ev:", ev)

    print("Ratio:", 1 - (ev / xH))
    print(time.time() - start)

    return xH

def corr_erd_ren(n, s, C):
    # creates a correlated Erdos-Renyi random graph from a random graph C
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
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

    ret = Decimal(math.factorial(n) / math.factorial(n - K - 1))
    ret *= Decimal(math.pow((p * s * s * (1 - p)), K))
    ret *= Decimal(0.5 * r * r * sizeT)
    return float(ret)

def calc_rec_Y(T, n, p, s, K, Corr):
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
    Y_corr = algorithm2(T, A_center, B_center, K)
    return Y_corr

def run_Y_comp(T, n, p, s, K, Corr, exp_corr):
    # run one time, get Y and compare
    # Corr is True when graphs are correlated, False when independent

    Y_corr = calc_rec_Y(T, n, p, s, K, Corr)

    print('rec_Y', Corr, Y_corr)
    #print('exp_Y', exp_corr)
    if Y_corr >= exp_corr:
        return 1
    else:
        return 0

def sim2(m, n, p, s, K):
    # runs simulation 2
    T = generateFreeTrees(K)
    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    exp_corr = calculateExpectedValueTwo(r, n, p, s, K, len(T))
    print(T)
    print(exp_corr)
    sum_corr = 0
    sum_ind = 0
    for i in range(m):
        sum_corr += run_Y_comp(T, n, p, s, K, True, exp_corr)
        sum_ind += run_Y_comp(T, n, p, s, K, False, exp_corr)
    sum_corr = sum_corr / m
    sum_ind = sum_ind / m
    return [sum_corr, sum_ind]

def kTiming(N,maxK):
    #function to test algo1 timings
    g = erd_ren(N,.4)
    timings = [[],[]]
    for x in range(1, maxK + 1):
        h = center_tree(x + 1)
        c = rand_assign(x,N)
        start = time.time()
        algorithmOne(h,g,c)
        end = time.time()
        timings[0].append(x)
        timings[1].append(end-start)
        print(timings)
    return timings

if __name__ == '__main__':

    #args1 = [3, 40, 0.5, [0, 1, 1, 1, 1, 1]]
    #simulation1(*args)

    # print(sim2(100, 99, 0.5, 0.7, 3))

    # print(time.time() - start)
    # kTiming(100,15)

    args2 = [20, 1000, 0.1, 0.8, 4]
    print(sim2(*args2))