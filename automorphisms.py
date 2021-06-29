"""
Created on 6/7/21
@author: louishu17, fayfayning, kieranlele
"""
import math

"""
checks if L1 and L2 are equal
"""
def checkL1L2(tree):
    if tree == tuple([0, 1]) or tree == [0, 1]:
        return True
    ind_1 = tree.index(1)
    m = tree.index(1, ind_1 + 1)
    L1 = [tree[i] - 1 for i in range(1, m)]
    L2 = [tree[i] for i in range(m, len(tree))]
    L2.insert(0, tree[0])
    if L1 == L2:
        return True
    return False


"""
Gets the label of the rth node of the tree
"""
def get_label(r, tree):
    lr = tree[r]
    r1 = len(tree)
    for i in range(r + 1, len(tree)):
        if tree[i] <= lr:
            r1 = i - 1
            break
    for j in range(r - 1, -1, -1):
        if tree[j] == lr - 1:
            r2 = j + 1
            break
    label = [r2]
    label.extend(tree[r:r1 + 1])
    return label

"""
Iterates through the tree to get all the labels
"""
def all_labels(tree):
    labels = []
    for r in range(1, len(tree)):
        label = get_label(r, tree)
        labels.append(label)
    return labels

"""
Calculates the number of automorphisms using the labels
"""
def calc_aut(labels):
    dict = {}
    num = 1
    for i in labels:
        tup = tuple(i)
        if tup not in dict:
            dict[tup] = 0
        dict[tup] += 1
    for i in dict.values():
        num *= math.factorial(i)
    return num

"""
calculates automorphisms based on a given tree
"""
def aut(tree):
    if tree == tuple([0, 1]) or tree == [0, 1]:
        return 2
    if not checkL1L2(tree):
        labels = all_labels(tree)
        num = calc_aut(labels)
    else:
        ind_1 = tree.index(1)
        m = tree.index(1, ind_1 + 1)
        L1 = [tree[i] - 1 for i in range(1, m)]
        labels = all_labels(L1)
        aut_L1 = calc_aut(labels)
        num = aut_L1 * aut_L1 * 2
    return num

