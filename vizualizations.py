"""
created a file for python vizualizations
trying out vscode functionality
new functionality with jupyter too
"""

import matplotlib.pyplot as plt
import time
import pandas as pd
from simulations import simulation1, calculateExpectedValueOne, sim2

def sim1_many():
    # runs simulation many times and outputs to a csv

    args = [3, False, 0.5, [0, 1, 1]]
    lst = [i * 10 for i in range(1, 51)]

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

        df.loc[len(df), :] = temp_lst

    df.to_csv('out1.csv', index=True, header=True)

def sim2_many():
    #m, n, p, s, K
    args = [20, 10, 0.5, 0.9, 3]
    lst = [10 ** i for i in range(1, 4)]

    df = pd.DataFrame(columns=['m', 'n', 'p', 's', 'K', 'corr', 'ind', 'time'])

    for i in lst:
        args[1] = i

        temp_lst = [i for i in args]

        start = time.time()
        res = sim2(*args)
        end = time.time() - start

        temp_lst.extend(res)
        temp_lst.append(end)

        df.loc[len(df), :] = temp_lst

    df.to_csv('out2.csv', index=True, header=True)

def scatter_sim1s(file):
    var1 = "n"
    var2 = "time"
    df = pd.read_csv(file)
    if var2 == "err":
        df["err"] = df["err"].abs()
    print(df.head(5))
    plt.scatter(x = df[var1], y = df[var2])
    plt.ylabel(var1)
    plt.xlabel(var2)
    plt.title("{} vs. {}".format(var1, var2))
    plt.show()


if __name__ == '__main__':
    #sim1_many()

    #file = "out1.csv"
    #scatter_sim1s(file)

    sim2_many()