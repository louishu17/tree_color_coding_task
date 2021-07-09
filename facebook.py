"""
Created on 6/28/21

@author: fayfayning
"""

import numpy as np
import pandas as pd
import math
import random
from simulations import alg2_fetch, erd_ren, calculateExpectedValueOne
from tree_generation import generateFreeTrees
from get_x import get_edges
from scipy.sparse import csr_matrix
from scipy import io
import os

def sample_n(A,B,n,p):
    x = random.randint(0,len(A) - n - 1)
    A = A[x:x + n,x:x + n]
    B = B[x:x + n,x:x + n]
    for i in range(n):
        for j in range(i):
            if A[i][j] == 1:
                temp = random.random()
                if temp < p:
                    continue
                else:
                    A[i][j] == 0
                    A[j][i] == 0
    
                
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

def tree_to_mat(tree):
    n = len(tree)
    edges = get_edges(tree)
    M = np.zeros((n, n))
    for i in edges:
        M[i[0] - 1][i[1] - 1] = 1
        M[i[1] - 1][i[0] - 1] = 1
    return M

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

def get_sim1_txt(n,p,file):
    graph = erd_ren(n,p)
    to_edge_list(graph,file)

def get_expected_list(top,increment,p,K,):
    ret = []
    for x in range(50, top, increment):
        ret.append(calculateExpectedValueOne(1,x,p,K))
    return ret

def convert_all_files(directory):
    save_path = '/Users/kieranlele/PycharmProjects/Data+Networks/tree_color_coding_task/facebook edge_lists'
    for filename in os.listdir(directory):
        new_name = filename[0:-4] + '.txt'
        completeName = os.path.join(save_path, new_name)  
        to_edge_list(read_mat('facebook100/' + filename),completeName)



if __name__ == '__main__':
    '''
    A = read_mat('American75.mat')
    B = read_mat('Amherst41.mat')
    m = 1
    ret_same = []
    ret_diff = []
    ret_same.append("same network")
    ret_diff.append("diff network")
    for x in range(0, m):
        ret.append(run_social_networks(A, B, 2, 100))
    print(ret)

    f1 = 'Auburn71.mat'
    to_edge_list(read_mat(f1), f1.split('.')[0] + '.txt')
    f1 = 'Baylor93.mat'
    to_edge_list(read_mat(f1), f1.split('.')[0] + '.txt')
    '''

    #print(tree_to_mat([0, 1, 2, 1]))
    '''
    get_sim1_txt(500,.8,'sim1_test_files/n_500.txt')
    '''
    '''
    print(get_expected_list(1050,50,.8,[0,1,2,2,1,2,2]))
    '''
    
    convert_all_files('/Users/kieranlele/PycharmProjects/Data+Networks/tree_color_coding_task/facebook100')

   # matrix = np.array([[0,1,1,1,1],[1,0,1,1,0],[1,1,0,1,0],[1,1,1,0,0],[1,0,
    # 0,0,0]])
    #print(to_edge_list(matrix, 'edge_list.txt'))
