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
    f = open(file, 'w')
    f.writelines(edge_list)
    f.close
    print(f)   


if __name__ == '__main__':
    
    A = read_mat('Auburn71.mat')
    
    B = read_mat('Baylor93.mat')
    print(to_edge_list(A))
    m = 2
    ret = []
    for x in range(0, m):
        ret.append(run_social_networks(A, B, 2, 100))
    print(ret)
    
    
    matrix = np.array([[0,1,1,1,1],[1,0,1,1,0],[1,1,0,1,0],[1,1,1,0,0],[1,0,0,0,0]])
    print(to_edge_list(A, 'edge_list.txt'))
    
    print(len(A))