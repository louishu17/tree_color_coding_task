"""
Created on 6/28/21

@author: fayfayning
"""

import numpy as np
import pandas as pd
import math
import random
from simulations import alg2_fetch
from tree_generation import generateFreeTrees
from get_x import get_edges
from scipy.sparse import csr_matrix
from scipy import io

def sample_n(A,B,n):
    x = random.randint(0,len(A) - n - 1)
    A = A[x:x + n,x:x + n]
    B = B[x:x + n,x:x + n]
    return [A,B]
def run_social_networks(A,B,K,n):
    samples = sample_n(A,B,n)
    T = generateFreeTrees(K)
    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    t = int(math.ceil(1 / (math.pow(r, 2))))
    return alg2_fetch(T, samples[0], samples[1], K, t)
def to_matrix(fname):
    file = open(fname)
    result = np.loadtxt(file, delimiter=",")
    return result
def read_mat(filename):
    loaded = io.loadmat(filename)
    data = loaded['A'].toarray()
    return data

def to_edge_list(matrix, file):
    edge_list = []
    edge_list.append(str((len(matrix))))
    edge_list.append(0)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if i < j:
                if matrix[i,j] == 1:
                    edge_list[1] += 1
                    string = "\n" + " ".join([str(i),str(j)])
                    edge_list.append(string)
            else:
                break
    edge_list[1] = "\n" + str((edge_list[1]))
    f = open(file, 'a')
    f.writelines(edge_list)
    f.close
    print(f)

def tree_to_mat(tree):
    n = len(tree)
    edges = get_edges(tree)
    M = np.zeros((n, n))
    for i in edges:
        M[i[0] - 1][i[1] - 1] = 1
        M[i[1] - 1][i[0] - 1] = 1
    return M



if __name__ == '__main__':
    '''
    A = read_mat('Auburn71.mat')
    B = read_mat('Baylor93.mat')
    print(to_edge_list(A))
    m = 2
    ret = []
    for x in range(0, m):
        ret.append(run_social_networks(A, B, 2, 100))
    print(ret)

    f1 = 'Auburn71.mat'
    to_edge_list(read_mat(f1), f1.split('.')[0] + '.txt')
    f1 = 'Baylor93.mat'
    to_edge_list(read_mat(f1), f1.split('.')[0] + '.txt')
    '''

    print(tree_to_mat([0, 1, 2, 1]))

   # matrix = np.array([[0,1,1,1,1],[1,0,1,1,0],[1,1,0,1,0],[1,1,1,0,0],[1,0,
    # 0,0,0]])
    #print(to_edge_list(matrix, 'edge_list.txt'))