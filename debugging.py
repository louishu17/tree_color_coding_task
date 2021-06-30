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

from get_x import get_edges, algorithmOne, rand_assign
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

if __name__ == '__main__':

    args = []
    args.append(int(sys.argv[1]))
    args.append(int(sys.argv[2]))
    args.append(float(sys.argv[3]))
    args.append(float(sys.argv[4]))
    args.append(int(sys.argv[5]))

    exh_sim2(*args)

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
import cProfile
import pstats

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

    # print(A_center)
    # print(B_center)
    Y_corr = alg2_fetch(T, A_center, B_center, K, t)
    return Y_corr


def run_Y_comp(T, n, p, s, K, Corr, exp_corr, t):
    # run one time, get Y and compare
    # Corr is True when graphs are correlated, False when independent

    # print("T:",len(T))

    Y_corr = calc_rec_Y(T, n, p, s, K, Corr, t)

    print(Y_corr)
    # print('exp_Y', exp_corr)
    if Y_corr >= exp_corr:
        return [Y_corr, 1]
    else:
        return [Y_corr, 0]


def sim2(m, n, p, s, K):
    # runs simulation 2
    T = generateFreeTrees(K)
    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    # t = int(math.ceil(1 / (math.pow(r, 2))))
    t = 1
    exp_corr = calculateExpectedValueTwo(r, n, p, s, K, len(T))
    print(T)
    print(exp_corr)
    sum_corr = 0
    sum_ind = 0
    # corr_vals = []
    # ind_vals = []
    print('Correlated')
    for i in range(m):
        corr = run_Y_comp(T, n, p, s, K, True, exp_corr, t)
        sum_corr += corr[1]
        # corr_vals.append(corr[0])
    print('Independent')
    for i in range(m):
        ind = run_Y_comp(T, n, p, s, K, False, exp_corr, t)
        sum_ind += ind[1]
        # ind_vals.append(ind[0])
    sum_corr = sum_corr / m
    sum_ind = sum_ind / m
    ret = [sum_corr, sum_ind]
    sys.stdout.flush()
    print(ret)
    return ret
    # return ["correlated", corr_vals, sum_corr, "independent", ind_vals, sum_ind]


def kTiming(N, maxK):
    # function to test algo1 timings
    g = erd_ren(N, .4)
    timings = [[], []]
    for x in range(1, maxK + 1):
        h = center_tree(x + 1)
        c = rand_assign(x, N)
        start = time.time()
        algorithmOne(h, g, c)
        end = time.time()
        timings[0].append(x)
        timings[1].append(end - start)
        print(timings)
    return timings


if __name__ == '__main__':
    args2 = []
    args2.append(int(sys.argv[1]))
    args2.append(int(sys.argv[2]))
    args2.append(float(sys.argv[3]))
    args2.append(float(sys.argv[4]))
    args2.append(int(sys.argv[5]))

    # args = [20, 100, 0.5, [0, 1, 2, 1]]
    # simulation1(*args)

    # print(sim2(100, 99, 0.5, 0.7, 3))

    # print(time.time() - start)
    # kTiming(100,15)

    # m, n, p, s, K
    # args2 = [2, 10, .1, 0.8, 3]
    start = time.time()
    sim2(*args2)
    end = time.time()
    print("Time:", end - start)

"""
Created on 6/7/21
@author: louishu17, fayfayning, kieranlele
"""
import cProfile
from itertools import combinations
import random
import multiprocessing as mp
import time
import math
import numpy as np
import sys
import os
import pstats

from tree_generation import generateFreeTrees

"""
Returns a list of all the edges in the tree
"""


def get_edges(tree):
    edges = []

    for i in range(1, len(tree)):
        edge = []

        # Finding R'- the parent node
        for j in range(i, -1, -1):
            if tree[j] == tree[i] - 1:
                # put in u_k in vertex in dict = R'
                edge.append(j + 1)

                break

        # put V_k in tertex
        edge.append(i + 1)

        edges.append(edge)

    return edges


"""
This method calcualtes the d for T_k
"""


def get_overcounting(tree):
    pieceset = set()

    first_1_ind = tree.index(1)

    try:
        second_1_ind = tree.index(1, first_1_ind + 1)
    except ValueError:
        return 1

    tup = tuple(tree[first_1_ind:second_1_ind])
    pieceset.add(tup)

    counter = 1

    left = second_1_ind

    for i in range(second_1_ind + 1, len(tree) + 1):
        if (i == len(tree) or tree[i] == 1):
            piece = tuple(tree[left:i])
            # print(tree, tup, piece)

            if piece in pieceset:
                counter += 1

            left = i

    return counter


"""
Splitting tree into Tak and Tbk
"""


def split_Trees(tree):
    first_1_ind = tree.index(1)

    second_exists = True
    try:
        second_1_ind = tree.index(1, first_1_ind + 1)
    except ValueError:
        second_exists = False

    if (second_exists):
        Tbk = [tree[i] - 1 for i in range(1, second_1_ind)]
        Tak = [tree[i] for i in range(second_1_ind, len(tree))]
        Tak.insert(0, 0)

        return (Tak, Tbk)
    else:
        Tbk = [tree[i] - 1 for i in range(1, len(tree))]
        Tak = [0]

        return (Tak, Tbk)


"""
Randomly Assigning colors (1, ..., K+1) to nodes 0 to n-1 inclusive nodes in G
"""


def rand_assign(K, n):
    all_colors = range(1, K + 2)
    colors = []
    for i in range(n):
        colors.append(random.choice(all_colors))
    return colors


"""
Generates a dictionary storing K,...,1 as keys, and T_k as the first value, d(T_k, V_k) as second value, and T_ak, and T_bk as 3rd and 4th value
"""


def get_Trees(tree_level, edges):
    # Edges list is indexed from 0, so edges[0] = edge 1

    # uk = r', and vk = k+1

    tree_dict = {}

    for k in range(len(edges), 0, -1):

        tree_dict.setdefault(k, [])

        r_double_prime = len(tree_level)

        # R is edges[k][1]-1
        for j in range(edges[k - 1][0], len(edges) + 1):

            if tree_level[j] <= tree_level[edges[k - 1][0] - 1] and tree_level[
                j - 1] > tree_level[edges[k - 1][0] - 1]:
                r_double_prime = j
                break

        T_k = []
        # from l_R+1 to l_R'' - l_R'
        T_k.append(0)
        for j in range(edges[k - 1][1], r_double_prime + 1):
            T_k.append(tree_level[j - 1] - tree_level[edges[k - 1][0] - 1])

        tree_dict[k].append(T_k)
        tree_dict[k].append(get_overcounting(T_k))

        # store T_ak, and T_bk
        T_ab = split_Trees(T_k)
        tree_dict[k].append(T_ab[0])
        tree_dict[k].append(T_ab[1])

    return tree_dict


"""
initializes all the X(x, T_0, c(x)), C is 0 indexed, however X_dict is 1 indexed
"""


def initialize_X(C, n):
    X_dict = {}

    for i in range(1, n + 1):
        zero_key = tuple([0])
        color_key = tuple([C[i - 1]])

        X_dict.setdefault(i, {}).setdefault(zero_key, {})[color_key] = 1

    return X_dict


"""
Finds combinations based on size k of C and memorizes it in color_dict
"""


def findSubsets(k, C, color_dict):
    color_dict.setdefault(k, [])
    colorCombos = list(combinations(C, k))
    color_dict[k].append(colorCombos)
    return colorCombos


def findc1c2Subsets(k, Cs, c1c2_dict):
    cs_key = tuple(Cs)
    c1c2_dict.setdefault(cs_key, {}).setdefault(k, [])
    colorCombos = list(combinations(Cs, k))
    c1c2_dict[cs_key][k] = colorCombos
    return colorCombos


"""
The X function in the research paper
"""


def X_func(X_dict, tree_dict, M, C, n, K, q):
    local_C = set(C)

    color_dict = {}
    c1c2_dict = {}

    # pprint.pprint(X_dict)

    for k in range(K, 0, -1):
        T_k = tuple(tree_dict[k][0])
        d = tree_dict[k][1]
        T_a = tuple(tree_dict[k][2])
        T_b = tuple(tree_dict[k][3])

        # print(T_k, d, T_a, T_b)

        if len(T_k) in color_dict:
            colorSubsets = color_dict[len(T_k)]
        else:
            colorSubsets = findSubsets(len(T_k), local_C, color_dict)

        for Cs in colorSubsets:
            Cs_key = tuple(Cs)

            t0 = time.time()

            # resultingSum is X(x, T_k, C) in paper
            resultingSum = 0

            if Cs_key in c1c2_dict:
                if len(T_b) in c1c2_dict[Cs_key]:
                    c1c2Subset = c1c2_dict[Cs_key][len(T_b)]
            else:
                c1c2Subset = findc1c2Subsets(len(T_b), Cs, c1c2_dict)

            # c1c2Subset = list(combinations(Cs, len(T_b)))
            # print(Cs, c1c2Subset)

            for x in range(1, n + 1):
                # print(c1c2Subset)

                outerSum = 0
                for y in range(1, n + 1):

                    if y == x:
                        continue

                    if (M[x - 1][y - 1] == 0):
                        continue

                    for i in c1c2Subset:
                        # set subtraction
                        c2 = set(i)
                        c1 = set(Cs) - c2

                        # dividing C into C1 and C2 wher C1 are the colors in Tak and C2 are the colors in Tbk
                        c1_key = tuple(c1)
                        c2_key = tuple(c2)

                        # try:
                        #     valueX = X_dict[x][T_a][c1_key]
                        # except KeyError:
                        #     valueX = "Not Found"

                        # try:
                        #     valueY = X_dict[y][T_b][c2_key]
                        # except KeyError:
                        #     valueY = "Not Found"

                        # print("X:", x, "Ta:", T_a, "C1", c1, "ValueX:", valueX)
                        # print("Y:", y, "Tb:", T_b, "C2", c2, "ValueY", valueY)
                        # print("outerSum:", outerSum)
                        # print()
                        try:
                            outerSum += (
                                    X_dict[x][T_a][c1_key] * X_dict[y][T_b][
                                c2_key] * M[x - 1][y - 1])
                        except KeyError:
                            continue

                # print(outerSum)
                resultingSum = outerSum / d
                X_dict.setdefault(x, {}).setdefault(T_k, {})[
                    Cs_key] = resultingSum

            # pprint.pprint(X_dict)

            t1 = time.time()
            # print("X_loop_time:", t1-t0)

    # pprint.pprint(X_dict)

    # XMH calculation
    finalSum = 0
    finalTreeKey = tuple(tree_dict[1][0])
    finalColorKey = tuple(list(range(1, K + 2)))
    for x in range(1, n + 1):
        try:
            finalSum += X_dict[x][finalTreeKey][finalColorKey]
        except:
            continue

    t1 = time.time()
    return finalSum / q


"""
This method returns q
"""


def check_equality(tree):
    if tree.count(1) < 2:
        return 2 if len(tree) == 2 else 1
    m = tree.index(1, 2)
    L1 = [tree[i] - 1 for i in range(1, m)]
    L2 = [tree[i] for i in range(m, len(tree))]
    L2.insert(0, 0)
    if L1 == L2:
        return 2
    return 1


"""
Algorithm one in research paper
"""


def algorithmOne(tree, M, C):
    t0 = time.time()
    edges = get_edges(tree)

    K = len(edges)
    n = len(M)

    finalSum = 0

    # print(C)
    X_dict = initialize_X(C, n)
    # print("Time Of Initialization:", t1-t0)

    # for keys, values in X_dict.items():
    #      print(keys, values)

    tree_dict = get_Trees(tree, edges)
    # print("Time of generating trees:", t2-t1)

    # for keys, values in tree_dict.items():
    #      print(keys, values)

    q = check_equality(tree)

    # print("Time of check equality:", t3-t2)

    # print("q", q)

    pr = cProfile.Profile()
    pr.enable()
    xMH = X_func(X_dict, tree_dict, M, C, n, K, q)
    pr.disable()
    # pstats.Stats(pr).print_stats()
    # print("Time of X_function:", t4-t3)

    return xMH


"""
Algorithm two in research paper
"""


def alg1_fetch(key, value, A, B, CA, CB):
    return value * algorithmOne(key, A, CA) * algorithmOne(key, B, CB)


def algorithm2(freeTrees, A, B, K):
    n = len(A)

    CA = rand_assign(K, n)
    CB = rand_assign(K, n)

    sumX = 0
    for keys, values in freeTrees.items():
        sumX += alg1_fetch(keys, values, A, B, CA, CB)
    return sumX


def alg2_fetch(freeTrees, A, B, K, t):
    sys.path.append(os.getcwd())
    try:
        npcus = int(os.environ['SLURM_JOB_CPUS_PER_TASK'])
    except KeyError:
        npcus = mp.cpu_count()
    pool = mp.Pool(npcus)
    results = pool.starmap(algorithm2, [(freeTrees, A, B, K) for i in range(t)])
    return sum(results) / t


"""
Generates erd_renyi graphs
"""


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


if __name__ == '__main__':
    t0 = time.time()

    A = np.ones((100, 100))

    B = np.ones((100, 100))

    K = 2

    # constant time
    freeTrees = generateFreeTrees(K)

    r = math.factorial(K + 1) / pow(K + 1, K + 1)

    t = math.ceil(1 / pow(r, 2))

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(algorithm2, [(freeTrees, A, B, K) for i in range(t)])
    pool.close()
    # ressum = 0
    # for i in range(t):
    #     ressum += algorithm2(freeTrees, A, B, K)

    resY = sum(results) / t
    t1 = time.time()
    print(resY)
    print("Time:", t1 - t0)

"""
Created on 6/7/21
@author: louishu17, fayfayning, kieranlele
"""
import math

"""
checks if L1 and L2 are equal
"""
def checkL1L2(tree):
    if tree == tuple([0, 1]) or tree == [0, 1]:
        return True
    ind_1 = tree.index(1)
    m = tree.index(1, ind_1 + 1)
    L1 = [tree[i] - 1 for i in range(1, m)]
    L2 = [tree[i] for i in range(m, len(tree))]
    L2.insert(0, tree[0])
    if L1 == L2:
        return True
    return False


"""
Gets the label of the rth node of the tree
"""
def get_label(r, tree):
    lr = tree[r]
    r1 = len(tree)
    for i in range(r + 1, len(tree)):
        if tree[i] <= lr:
            r1 = i - 1
            break
    for j in range(r - 1, -1, -1):
        if tree[j] == lr - 1:
            r2 = j + 1
            break
    label = [r2]
    label.extend(tree[r:r1 + 1])
    return label

"""
Iterates through the tree to get all the labels
"""
def all_labels(tree):
    labels = []
    for r in range(1, len(tree)):
        label = get_label(r, tree)
        labels.append(label)
    return labels

"""
Calculates the number of automorphisms using the labels
"""
def calc_aut(labels):
    dict = {}
    num = 1
    for i in labels:
        tup = tuple(i)
        if tup not in dict:
            dict[tup] = 0
        dict[tup] += 1
    for i in dict.values():
        num *= math.factorial(i)
    return num

"""
calculates automorphisms based on a given tree
"""
def aut(tree):
    if tree == tuple([0, 1]) or tree == [0, 1]:
        return 2
    if not checkL1L2(tree):
        labels = all_labels(tree)
        num = calc_aut(labels)
    else:
        ind_1 = tree.index(1)
        m = tree.index(1, ind_1 + 1)
        L1 = [tree[i] - 1 for i in range(1, m)]
        labels = all_labels(L1)
        aut_L1 = calc_aut(labels)
        num = aut_L1 * aut_L1 * 2
    return num


