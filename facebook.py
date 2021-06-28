"""
Created on 6/28/21

@author: fayfayning
"""

import numpy as np
import pandas as pd
import math
from simulations import alg2_fetch
from tree_generation import generateFreeTrees

if __name__ == '__main__':
    df = pd.read_csv('American75.csv')
    A = df.to_numpy()
    df = pd.read_csv('Amherst41.csv')
    B = df.to_numpy()
    K = 4
    T = generateFreeTrees(K)
    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    t = int(math.ceil(1 / (math.pow(r, 2))))
    print(alg2_fetch(T, A, B, K, t))