"""
Created on 6/7/21
@author: louishu17, fayfayning, kieranlele
"""
import math
import time
import random
import sys
import decimal
import multiprocessing as mp
import numpy as np
import os

from automorphisms import aut
from get_x import algorithmOne, alg2_fetch, rand_assign
from tree_generation import generateFreeTrees, center_tree

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


def erd_ren_centered(n, p):

    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            temp = random.random()
            if temp < p:
                M[i][j] = 1 - p
                M[j][i] = M[i][j]
            else:
                M[i][j] = 0 - p
                M[j][i] = M[i][j]
    

    return M


def calculateExpectedValueOne(m, n, p, H):
    # returns expected value based on r, n, K, p, H
    print(m, n, p, H)
    r = decimal.Decimal(math.factorial(len(H)) / math.pow(len(H), len(H)))
    exp_val = decimal.Decimal(r * math.factorial(n) / math.factorial(n - len(
        H)))
    exp_val *= decimal.Decimal(math.pow(p, len(H) - 1) / aut(H))
    return float(exp_val)


def simulation1(m, n, p, H):
    # runs simulation 1
    start = time.time()

    K = len(H) - 1

    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)

    t = int(math.ceil(1 / (math.pow(r, 2))))

    # ev = calculateExpectedValueOne(m, n, p, H)

    sys.path.append(os.getcwd())
    try:
        ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    except KeyError:
        ncpus = mp.cpu_count()
    pool = mp.Pool(ncpus)
    graphs = pool.starmap(erd_ren_centered, [(n, p) for i in range(m)])

    resMSum = 0
    for g in graphs:
        print()
        print(g)
        color_nodes = pool.starmap(rand_assign, [(K, n) for i in range(t)])
        results = pool.starmap(algorithmOne, [(H, g, C) for C in color_nodes])
        xH = sum(results) / t
        print(xH)
        resMSum += xH

    pool.close()

    avgxH = resMSum / m
    print("xH:", avgxH)

    print(time.time() - start)

    return avgxH

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

    ret = decimal.Decimal(math.factorial(n) / math.factorial(n - K - 1))
    ret *= decimal.Decimal(math.pow((p * s * s * (1 - p)), K))
    ret *= decimal.Decimal(0.5 * r * r * sizeT)
    return float(ret)

def calc_rec_Y(T, n, p, s, K, Corr, t):
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
    Y_corr = alg2_fetch(T, A_center, B_center, K, t)
    return Y_corr

def run_Y_comp(T, n, p, s, K, Corr, exp_corr, t):
    # run one time, get Y and compare
    # Corr is True when graphs are correlated, False when independent

    #print("T:",len(T))

    Y_corr = calc_rec_Y(T, n, p, s, K, Corr, t)

    print(Y_corr)
    #print('exp_Y', exp_corr)
    if Y_corr >= exp_corr:
        return [Y_corr, 1]
    else:
        return [Y_corr, 0]

def sim2(m, n, p, s, K):
    # runs simulation 2
    T = generateFreeTrees(K)
    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    t = int(math.ceil(1 / (math.pow(r, 2))))
    exp_corr = calculateExpectedValueTwo(r, n, p, s, K, len(T))
    print(T)
    print(exp_corr)
    sum_corr = 0
    sum_ind = 0
    #corr_vals = []
    #ind_vals = []
    print('Correlated')
    for i in range(m):
        corr = run_Y_comp(T, n, p, s, K, True, exp_corr, t)
        sum_corr += corr[1]
        #corr_vals.append(corr[0])
    print('Independent')
    for i in range(m):
        ind = run_Y_comp(T, n, p, s, K, False, exp_corr, t)
        sum_ind += ind[1]
        #ind_vals.append(ind[0])
    sum_corr = sum_corr / m
    sum_ind = sum_ind / m
    ret = [sum_corr, sum_ind]
    sys.stdout.flush()
    print(ret)
    return ret
    #return ["correlated", corr_vals, sum_corr, "independent", ind_vals, sum_ind]

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
    args2 = []
    args2.append(int(sys.argv[1]))
    args2.append(int(sys.argv[2]))
    args2.append(float(sys.argv[3]))
    args2.append(float(sys.argv[4]))
    args2.append(int(sys.argv[5]))

    #args = [20, 100, 0.5, [0, 1, 2, 1]]
    #simulation1(*args)

    # print(sim2(100, 99, 0.5, 0.7, 3))

    # print(time.time() - start)
    # kTiming(100,15)
    
    #m, n, p, s, K
    #args2 = [2, 10, .1, 0.8, 3]
    start = time.time()
    sim2(*args2)
    end = time.time()
    print("Time:", end-start)