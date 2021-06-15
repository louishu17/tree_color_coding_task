"""
Created on 6/7/21
@author: louishu17, fayfayning, kieranlele
"""
import math
import time

from automorphisms import aut
from get_x import getX, algorithm2
from tree_generation import generateFreeTrees

import multiprocessing as mp
from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def generateErdosReyniGraphs(n, p, m):
    result = []
    for i in range(m):
        G = nx.generators.random_graphs.gnp_random_graph(n, p)
        A = nx.adjacency_matrix(G)
        result.append(A)

    return result


def calculateExpectedValueOne(r, n, K, p, L):
    return r * ((math.factorial(n)) / (
                math.factorial(n - K - 1) * aut(L))) * pow(p, K)


def simulation1():
    start = time.time()

    n = 1000
    p = 0.3
    m = 1

    graphs = generateErdosReyniGraphs(n, p, m)

    L = [0, 1, 1]

    K = len(L) - 1

    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)

    t = int(math.ceil(1 / (math.pow(r, 2))))

    ev = calculateExpectedValueOne(r, n, K, p, L)

    resMSum = 0
    for g in graphs:
        a = g.todense().tolist()
        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(getX, [(L, a, K, n) for i in range(t)])
        pool.close()

        resMSum += sum(results) / t

    xH = resMSum / m

    print("Ratio:", 1 - (ev / xH))
    print(time.time() - start)


def generateErdosReyniGraphEdges(n, p):
    result = []
    G = nx.generators.random_graphs.gnp_random_graph(n, p)
    A = list(G.edges())
    # print(A)
    # nx.draw(G, with_labels = True)
    # plt.show()
    result.append(A)

    return result


def sEdgeSelection(edges, s):
    ps = np.random.binomial(size=len(edges), n=1, p=s)
    for i in range(len(edges) - 1, -1, -1):
        if ps[i] == 0:
            del edges[i]
    print(edges)

    return edges


def edgesToAdjMatrix(edges, n):
    adjacency_matrix = [[0 for x in range(n)] for y in range(n)]
    if len(edges) != 0:
        for i, j in edges[0]:
            adjacency_matrix[i][j] = 1

    return adjacency_matrix


def centerAdjMatrix(g, n, p, s):
    for i in range(0, n):
        for j in range(0, i):
            g[i][j] = g[i][j] - p * s
            g[j][i] = g[i][j]

    return g

def calculateExpectedValueTwo(r, n, p, s, K, sizeT):
    return 0.5 * pow(r, 2) * ((math.factorial(n) * pow(
        (p * pow(s, 2) * (1 - p)), K)) / math.factorial(n - K - 1)) * sizeT

def simulation2():

    n = 10
    p = 0.3
    m = 2
    s = 0.75

    K = 7

    freeTrees = generateFreeTrees(K)

    r = math.factorial(K+1) / pow(K+1, K+1)

    t = math.ceil(1/pow(r,2))

    ev = calculateExpectedValueTwo(r, n, p, s, K, len(freeTrees))

    for i in range(m):
        edgesA = generateErdosReyniGraphEdges(n, p)
        edgesB = generateErdosReyniGraphEdges(n, p)

        sEdgesA = sEdgeSelection(edgesA, s)
        sEdgesB = sEdgeSelection(edgesB, s)

        graphA = edgesToAdjMatrix(sEdgesA, n)
        graphB = edgesToAdjMatrix(sEdgesB, n)

        centeredGraphA = centerAdjMatrix(graphA, n, p, s)
        centeredGraphB = centerAdjMatrix(graphB, n, p, s)




        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(algorithm2, [(freeTrees, centeredGraphA, centeredGraphB, K) for i in range(t)])
        pool.close()
        resY = sum(results) / t
        print(resY)

        if resY >= ev:
            print(1)
        else:
            print(0)

if __name__ == '__main__':