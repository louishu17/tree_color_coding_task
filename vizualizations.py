"""
created a file for python vizualizations
trying out vscode functionality
new functionality with jupyter too
"""

import pandas as pd
from simulations import simulation1, calculateExpectedValueOne

def sim1_many():
    # runs simulation many times and outputs to a csv

    args = [3, False, 0.5, [0, 1, 1, 1, 1]]
    lst = [i * 10 for i in range(1, 51)]

    df = pd.DataFrame(columns=['m', 'n', 'p', 'H', 'K', 'rec', 'exp', 'err'])

    try:
        for i in lst:
            args[1] = i

            temp_lst = args[:3]
            str_tree = '-'.join([str(i) for i in args[3]])
            temp_lst.append(str_tree)
            temp_lst.append(len(args[3]) - 1)

            rec_val = simulation1(*args)
            ev = calculateExpectedValueOne(*args)
            err = 1 - (ev / rec_val)

            temp_lst.append(rec_val)
            temp_lst.append(ev)
            temp_lst.append(err)

            df.loc[len(df), :] = temp_lst
    except OverflowError:
        print('OverflowError: int too large to convert to float')
    df.to_csv('out.csv', index=True, header=True)

if __name__ == '__main__':
    sim1_many()