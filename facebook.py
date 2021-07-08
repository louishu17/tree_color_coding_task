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
from scipy.sparse import csr_matrix
from scipy import io

def sample_n(A,B,n):
    x = random.randint(0,len(A) - n - 1)
    A = A[x:x + n,x:x + n]
    B = B[x:x + n,x:x + n]
    return [A,B]

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
    f = open(file, 'w')
    f.writelines(edge_list)
    f.close
    print(f)   

def subsample_run(A,K,n):
    samples = sample_n(A,A,n)
    T = generateFreeTrees(K)
    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    t = int(math.ceil(1 / (math.pow(r, 2))))
    return alg2_fetch(T, samples[0], samples[1], K, t)

def run_two_networks(A,B,K,n):
    samples = sample_n(A,B,n)
    T = generateFreeTrees(K)
    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    t = int(math.ceil(1 / (math.pow(r, 2))))
    return alg2_fetch(T, samples[0], samples[1], K, t)


if __name__ == '__main__':
    
    A = read_mat('American75.mat')
    B = read_mat('Amherst41.mat')
    m = 1
    ret_same = []
    ret_diff = []
    ret_same.append("same network")
    ret_diff.append("diff network")
    for x in range(0, m):
        ret_diff.append(run_two_networks(A, B, 5, 100))
        ret_same.append(subsample_run(A,5,1000))
    print(ret_diff,ret_same)
    