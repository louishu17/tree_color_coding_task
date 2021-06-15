"""
Created on 6/7/21
@author: louishu17, fayfayning, kieranlele
"""
from automorphisms import aut

"""
This method finds the primary root of an n-path tree
"""


def center_tree(n):
    tree = range(n)
    if len(tree) % 2 == 1:
        count = int((len(tree) - 1) / 2)
        new_tree = [i for i in range(0, count + 1)]
        new_tree.extend(range(1, count + 1))
    else:
        count = int(len(tree) / 2)
        new_tree = [i for i in range(0, count + 1)]
        new_tree.extend(range(1, count))
    return new_tree


"""
This method finds the p and q for the sucessor function in the research paper of given tree
"""


def find_pq(tree):
    for i in range(len(tree) - 1, -1, -1):
        if tree[i] != 1:
            p = i
            lp = tree[i]
            break
    for i in range(p - 1, -1, -1):
        if tree[i] == lp - 1:
            return [p + 1, i + 1]


"""
Successor Function
"""


def get_si(tree, vals):
    output = []

    # i is index
    i = 1
    while (i <= len(tree)):
        if (i < vals[0]):
            # array is 0 indexed so we have to subtract 1 in order to get right value
            # l_i
            output.append(tree[i - 1])
        else:
            # s_i-(p-q)
            output.append(output[i - 1 - (vals[0] - vals[1])])
        i += 1
    return output


"""
Wright, Richmond, Odlyzko and McKay check free tree algorithm
"""


def free_check(tree, n):
    # index of first 1 in tree
    ind_1 = tree.index(1)

    # index of second 1 in tree, we guarantee a second one due the fact that we start with a n-path rooted at its primary root
    m = tree.index(1, ind_1 + 1)

    # L1 and L2 split based on m--the second one
    L1 = [tree[i] - 1 for i in range(1, m)]
    L2 = [tree[i] for i in range(m, n)]
    L2.insert(0, tree[0])

    # we already guarantee condition (a) so we just have to check (b), (c), (d)
    # condition (b) in paper
    if max(L2) < max(L1):
        return False
    # condition (c)
    if max(L2) == max(L1):
        if len(L1) > len(L2):
            return False
    # condition (d)
    if len(L1) == len(L2):
        if L1 > L2:
            return False
    return True


"""
"Jump" function when condition fails
"""


def skip(tree):
    ind_1 = tree.index(1)

    m = tree.index(1, ind_1 + 1)

    # finding p and q

    # p' is the last node of the L1
    pqArr = [0, 0]

    pqArr[0] = m - 1

    # finding new q
    for i in range(m - 1, -1, -1):
        if tree[i] == tree[pqArr[0]] - 1:
            # have to add one due to RESEARCH paper indexing from 1
            pqArr[1] = i + 1
            pqArr[0] += 1
            break

    tree2 = get_si(tree, pqArr)
    ind_1_prime = tree2.index(1)
    try:
        m_prime = tree2.index(1, ind_1_prime + 1)
    except ValueError:
        m_prime = len(tree)
    L1_prime = [tree2[i] - 1 for i in range(1, m_prime)]

    # subtract one due to adding of one previously to look at right index
    if tree[pqArr[0] - 1] > 2:
        h = max(L1_prime)
        tree2 = tree2[:len(tree2) - (h + 1)]
        tree2.extend(range(1, h + 2))

    return tree2


"""
Genereates all the free trees...
"""


def generateFreeTrees(K):
    # adding initial primary rooted free tree
    free = {}
    tree = center_tree(K)
    tree_key = tuple(tree)
    free.setdefault(tree_key, aut(tree))

    # loops through until it hits base case of [0, 1, 1,..., 1]
    while sum(tree) != len(tree) - 1:
        vals = find_pq(tree)
        tree = get_si(tree, vals)
        while (free_check(tree, K) == False):
            tree = skip(tree)

        tree_key = tuple(tree)
        free.setdefault(tree_key, aut(tree))

    return free

if __name__ == '__main__':
