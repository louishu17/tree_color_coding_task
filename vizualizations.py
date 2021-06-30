"""
created a file for python vizualizations
trying out vscode functionality
new functionality with jupyter too
"""

import math
import matplotlib.pyplot as plt
import time
import pandas as pd
from simulations import calculateExpectedValueTwo, simulation1, calculateExpectedValueOne, sim2, calculateExpectedValueTwo, calc_rec_Y
from tree_generation import generateFreeTrees
from exhaustive import calc_rec_Y_both


def sim1_many():
    # runs simulation many times and outputs to a csv
    # m, n, p, H
    args = [3, False, 0.5, [0, 1, 2, 1, 1]]
    lst = [i * 50 for i in range(6, 9)]

    """
    for i in range(len(lst)):
        lst[i].extend([1 for j in range(i + 2)])
    print(lst)
    """

    df = pd.DataFrame(columns=['m', 'n', 'p', 'H', 'K', 'rec', 'exp', 'err',
                               'time'])

    for i in lst:
        args[1] = i

        temp_lst = args[:3]
        str_tree = '-'.join([str(i) for i in args[3]])
        temp_lst.append(str_tree)
        temp_lst.append(len(args[3]) - 1)

        start = time.time()
        rec_val = simulation1(*args)
        end = time.time() - start

        ev = calculateExpectedValueOne(*args)
        err = 1 - (ev / rec_val)

        temp_lst.append(rec_val)
        temp_lst.append(ev)
        temp_lst.append(err)
        temp_lst.append(end)

        print('temp_list', temp_lst)

        df.loc[len(df), :] = temp_lst

    df.to_csv('out1.csv', index=True, header=True)

def sim2_many():
    #m, n, p, s, K
    args = [20, 100, 0.5, False, 3]
    lst = [0.1 * i for i in range(6, 10)]

    df = pd.DataFrame(columns=['m', 'n', 'p', 's', 'K', 'corr', 'ind', 'time'])

    for i in range(len(args)):
        if args[i] == False:
            alt = i

    for i in lst:
        args[alt] = i

        temp_lst = [i for i in args]

        start = time.time()
        print()
        print(i)
        res = sim2(*args)
        end = time.time() - start

        temp_lst.extend(res)
        temp_lst.append(end)

        df.loc[len(df), :] = temp_lst

    df.to_csv('out2.csv', index=True, header=True)

def scatter_simp(file):
    var1 = 'n'
    var2 = "time"
    df = pd.read_csv(file)
    if var2 == "err":
        df["err"] = df["err"].abs()
    print(df.head(5))
    plt.scatter(x=df[var1], y=df[var2], label=var2)
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title("{} vs. {}".format(var1, var2))
    plt.show()

def scatter_corr_ind(file):
    var1 = 'p'
    var2 = "corr"
    var3 = "ind"
    df = pd.read_csv(file)
    print(df.head(5))
    plt.scatter(x = df[var1], y = df[var2], color='green',label=var2)
    plt.scatter(x = df[var1], y = df[var3], color='blue',label=var3)
    plt.legend()
    plt.xlabel(var1)
    plt.ylabel("Proportion")
    plt.title("{} vs. {} and {}".format(var1, "ind", var2))
    plt.show()

def mass_test_sim2(Corr):
    # test
    m = 1
    n = 100
    K = 8
    p = .05
    s = 1
    rec_Y_vals = []
    T = generateFreeTrees(K)
    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    t = 1
    exp_Y = calculateExpectedValueTwo(r, n, p, s, K, len(T))
    comp = []
    for i in range(m):
        rec_Y_temp = calc_rec_Y(T, n, p, s, K, Corr, t)
        rec_Y_vals.append(rec_Y_temp)
        print('rec_Y_vals', rec_Y_vals)
        if rec_Y_temp >= exp_Y:
            comp.append(1)
        else:
            comp.append(0)
    print('exp_Y', exp_Y)
    print('average rec_Y', sum(rec_Y_vals) / m)
    print('exceeds expected', comp)
    print('proportion', sum(comp) / m)


def test_both(Corr):
    # test
    m = 1
    n = 10
    K = 2
    p = .05
    s = 1
    rec_Y_exhaust = []
    rec_Y_algo = []
    T = 1
    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    t = 1
    exp_y = calculateExpectedValueTwo(r, n, p, s, K, len(T))
    for i in range(m):
        rec_Y_temp = calc_rec_Y_both(T, n, p, s, K, Corr, t)
        rec_Y_exhaust.append(rec_Y_temp[1])
        rec_Y_algo.append(rec_Y_temp[0])
        print('rec_Y_vals', rec_Y_temp, rec_Y_algo)
    print('exp_Y', exp_y)

if __name__ == '__main__':
    #sim2_many()

    #file = "out1.csv"
    #scatter_simp(file)

    #sim2_many()

    #file = "out2.csv"
    #scatter_simp(file)

    mass_test_sim2(True)
    mass_test_sim2(False)