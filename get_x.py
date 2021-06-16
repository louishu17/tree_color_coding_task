"""
Created on 6/7/21
@author: louishu17, fayfayning, kieranlele
"""
import itertools
import random

import multiprocessing as mp
import time
import math
from tree_generation import generateFreeTrees
import networkx as nx
import pprint




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

    tree_dict = {}

    for k in range(len(edges), 0, -1):

        tree_dict.setdefault(k, [])

        r_double_prime = len(tree_level)

        # R is edges[k][1]-1
        for j in range(edges[k - 1][0], len(edges) + 1):

            if tree_level[j] == tree_level[edges[k - 1][0] - 1]:
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
        tree_dict[k].append(split_Trees(T_k)[0])
        tree_dict[k].append(split_Trees(T_k)[1])

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

    
def X_func(X_dict, tree_dict, M, C, n, K, q):

    local_C = set(C)

    t0 = time.time()
    for k in range(K, 0, -1):
        print()

        print(k)
        T_k = tuple(tree_dict[k][0])
        d = tree_dict[k][1]
        T_a = tuple(tree_dict[k][2])
        T_b = tuple(tree_dict[k][3])

        # print(T_k, d, T_a, T_b)

        t5 = time.time()
        colorSubsets = list(itertools.combinations(local_C, len(T_k)))
        t6 = time.time()
        print(colorSubsets)

        # print("Time to find color subsets:", t6-t5)
        colors = []

        t3 = time.time()
        for Cs in colorSubsets:

                c1c2Subset = list(itertools.combinations(Cs, len(T_b)))

                for i in c1c2Subset:

                    # set subtraction
                    c2 = set(i)
                    c1 = set(Cs) - c2

                    # print(Cs, c1, c2)

                    if len(c1) < len(T_a) or len(c2) < len(T_b):
                        # print("SKIPPED")
                        continue

                    colors.append((tuple(c1),tuple(c2)))

        t4 = time.time()

        # print("Generating Color Subsets c1 and c2 Time:", t4-t3)
        # print(colors)
                    

        
        for x in range(1, n + 1):
            for Cs in colorSubsets:
                # print(x)
                t0 = time.time()
                
                # resultingSum is X(x, T_k, C) in paper
                resultingSum = 0

                outerSum = 0
                for y in range(1, n + 1):

                    if y == x:
                        continue

                    if (M[x - 1][y - 1] == 0):
                        continue

                    for c1_key, c2_key in colors:
                        innerSum = 0

                        # dividing C into C1 and C2 wher C1 are the colors in Tak and C2 are the colors in Tbk
                        
                        try:
                            innerSum += (
                                        X_dict[x][T_a][c1_key] * X_dict[y][T_b][
                                    c2_key] * M[x - 1][y - 1])
                        except KeyError:
                            continue

                    outerSum += innerSum

                resultingSum = outerSum / d
                Cs_key = tuple(Cs)
                X_dict.setdefault(x, {}).setdefault(T_k, {})[
                    Cs_key] = resultingSum

                t1 = time.time()
                # print("X_loop_time:", t1-t0)
    
    

    # pprint.pprint(X_dict)

    #XMH calculation
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


def algorithmOne(tree, M, C):
    t0 = time.time()
    edges = get_edges(tree)

    K = len(edges)
    n = len(M)

    finalSum = 0

    # print(C)
    X_dict = initialize_X(C, n)

    t1 = time.time()
    print("Time Of Initialization:", t1-t0)

    # for keys, values in X_dict.items():
    #      print(keys, values)

    tree_dict = get_Trees(tree, edges)
    t2 = time.time()
    print("Time of generating trees:", t2-t1)

    # for keys, values in tree_dict.items():
    #      print(keys, values)


    q = check_equality(tree)
    t3 = time.time()

    print("Time of check equality:", t3-t2)


    # print("q", q)

    finalSum = X_func(X_dict, tree_dict, M, C, n, K, q)
    t4 = time.time()

    print("Time of X_function:", t4-t3)

    return finalSum



def algorithm2(freeTrees, A, B, K):
    sumX = 0
    n = len(A)
    CA = rand_assign(K, n)
    CB = rand_assign(K, n)
    for keys, values in freeTrees.items():
        sumX += (values * algorithmOne(keys, A, CA) * algorithmOne(keys, B, CB))
    return sumX


def generateErdosReyniGraph(n, p):

    G = nx.generators.random_graphs.gnp_random_graph(n, p)
    A = nx.adjacency_matrix(G)
    return A



if __name__ == '__main__':


    # A = [[0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    #      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #      [1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    #      [0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
    #      [1, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    #      [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    #      [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    #      [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]]
    
    # B = [[0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    #      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #      [1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    #      [0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
    #      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    #      [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #      [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]]

    A = generateErdosReyniGraph(100,0.4).todense().tolist()
    L = [0, 1, 2, 1, 2]
    C = rand_assign(len(L)-1, len(A))
    trials = 1
    times = []
    for i in range(trials):
        t0 = time.time()
        a = algorithmOne(L, A, C)
        t1 = time.time()
        times.append(t1-t0)
    print(a)
    print("Average Time:", sum(times)/len(times))

    

    # K = 4

    # #constant time
    # freeTrees = generateFreeTrees(K)


    # r = math.factorial(K+1) / pow(K+1, K+1)

    # t = math.ceil(1/pow(r,2))


    # pool = mp.Pool(mp.cpu_count())
    # results = pool.starmap(algorithm2, [(freeTrees, A, B, K) for i in range(t)])
    # pool.close()
    # # ressum = 0
    # # for i in range(t):
    # #     ressum += algorithm2(freeTrees, A, B, K)

    # resY = sum(results) / t
    # t1 = time.time() 
    # print(resY)
    # print("Time:", t1-t0)


