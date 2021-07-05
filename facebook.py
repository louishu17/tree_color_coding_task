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
import csv

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
if __name__ == '__main__':
    A = to_matrix('American75.csv')
    B = to_matrix('Amherst41.csv')
    m = 2
    ret = []
    for x in range(0,m):
        ret.append(run_social_networks(A,B,2,100))
    print(ret)