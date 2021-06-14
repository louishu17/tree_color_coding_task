"""
Created on 6/7/21
@author: louishu17, fayfayning, kieranlele
"""

import random
import itertools
import time
import math

import multiprocessing as mp

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
        new_tree = [i for i in range(0, count+1)]
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
Successor Funciton
"""
def get_si(tree, vals):
    output = []


    #i is index
    i = 1
    while(i <= len(tree)):
        if(i < vals[0]):
            #array is 0 indexed so we have to subtract 1 in order to get right value
            #l_i
            output.append(tree[i-1])
        else:
            #s_i-(p-q)
            output.append(output[i-1-(vals[0]-vals[1])])
        i += 1
    return output

"""
Wright, Richmond, Odlyzko and McKay check free tree algorithm
"""
def free_check(tree, n):

    #index of first 1 in tree
    ind_1 = tree.index(1)

    #index of second 1 in tree, we guarantee a second one due the fact that we start with a n-path rooted at its primary root
    m = tree.index(1, ind_1 + 1)

    #L1 and L2 split based on m--the second one
    L1 = [tree[i] - 1 for i in range(1, m)]
    L2 = [tree[i] for i in range(m, n)]
    L2.insert(0, tree[0])


    #we already guarantee condition (a) so we just have to check (b), (c), (d)
    #condition (b) in paper
    if max(L2) < max(L1):
        return False
    #condition (c)
    if max(L2) == max(L1):
        if len(L1) > len(L2):
            return False
    #condition (d)
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


    #finding p and q

    #p' is the last node of the L1
    pqArr = [0, 0]

    pqArr[0] =  m - 1

    #finding new q
    for i in range(m - 1, -1, -1):
        if tree[i] == tree[pqArr[0]] - 1:

            #have to add one due to RESEARCH paper indexing from 1
            pqArr[1] = i + 1
            pqArr[0] += 1
            break
    

    tree2 = get_si(tree, pqArr)
    ind_1_prime = tree2.index(1)
    try:
        m_prime =  tree2.index(1, ind_1_prime + 1)
    except ValueError:
        m_prime = len(tree)
    L1_prime = [tree2[i] - 1 for i in range(1, m_prime)]


    #subtract one due to adding of one previously to look at right index
    if tree[pqArr[0]-1] > 2:
        h = max(L1_prime)
        tree2 = tree2[:len(tree2) - (h + 1)]
        tree2.extend(range(1, h + 2))

    return tree2


"""
Genereates all the free trees...
"""
def generateFreeTrees(K):
    #adding initial primary rooted free tree
    free = {}
    tree = center_tree(K)
    tree_key = tuple(tree)
    free.setdefault(tree_key, aut(tree))


    #loops through until it hits base case of [0, 1, 1,..., 1]
    while sum(tree) != len(tree) - 1:
        vals = find_pq(tree)
        tree = get_si(tree, vals)
        while(free_check(tree, K) == False):
            tree = skip(tree)

        tree_key = tuple(tree)
        free.setdefault(tree_key, aut(tree))

    return free


"""
checks if L1 and L2 are equal
"""
def checkL1L2(tree):
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
Runs program on a given tree
"""
def aut(tree):
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


"""
Returns a list of all the edges in the tree
"""
def get_edges(tree):

    edges = [] 
    
    for i in range(1, len(tree)):
        edge = []

        #Finding R'- the parent node
        for j in range(i, -1, -1):
            if tree[j] == tree[i] - 1:
                 #put in u_k in vertex in dict = R'
                edge.append(j+1)

                break

        #put V_k in tertex
        edge.append(i+1)

        edges.append(edge)

    return edges


"""
This method calcualtes the d for T_k
"""

def get_overcounting(tree):
    pieceset = set()

    first_1_ind = tree.index(1)
    
    try:
        second_1_ind = tree.index(1, first_1_ind+1)
    except ValueError:
        return 1
     
    tup = tuple(tree[first_1_ind:second_1_ind])
    pieceset.add(tup)

    counter = 1

    left = second_1_ind

    for i in range(second_1_ind+1, len(tree)+1):
        if(i == len(tree) or tree[i] == 1):
            piece = tuple(tree[left:i])
            # print(tree, tup, piece)

            if piece in pieceset:
                counter += 1

            left = i
        

    return counter

"""
Splitting tree into Tak and Tbk
"""

def split_Trees(tree):

    first_1_ind = tree.index(1)

    second_exists = True
    try:
        second_1_ind = tree.index(1, first_1_ind+1)
    except ValueError:
        second_exists = False
    
    if(second_exists):
        Tbk = [tree[i] - 1 for i in range(1,second_1_ind)]
        Tak = [tree[i] for i in range(second_1_ind,len(tree))]
        Tak.insert(0,0)

        return (Tak, Tbk)
    else:
        Tbk = [tree[i]-1 for i in range(1, len(tree))]
        Tak = [0]

        return (Tak, Tbk)

"""
Randomly Assigning colors (1, ..., K+1) to nodes 0 to n-1 inclusive nodes in G
"""

def rand_assign(K, n):
    all_colors = range(1, K+2)
    colors = []
    for i in range(n):
        colors.append(random.choice(all_colors))
    return colors


"""
Generates a dictionary storing K,...,1 as keys, and T_k as the first value, d(T_k, V_k) as second value, and T_ak, and T_bk as 3rd and 4th value
"""
def get_Trees(tree_level, edges):

    #Edges list is indexed from 0, so edges[0] = edge 1

    tree_dict = {}


    for k in range(len(edges), 0, -1):

        tree_dict.setdefault(k, [])

        r_double_prime = len(tree_level) 

        #R is edges[k][1]-1
        for j in range(edges[k-1][0], len(edges)+1):

            if tree_level[j] == tree_level[edges[k-1][0]-1]:
                r_double_prime = j
                break

        T_k = []
        #from l_R+1 to l_R'' - l_R'
        T_k.append(0)
        for j in range(edges[k-1][1], r_double_prime+1):
            T_k.append(tree_level[j-1] - tree_level[edges[k-1][0]-1])
        

        tree_dict[k].append(T_k)
        tree_dict[k].append(get_overcounting(T_k))
        
        #store T_ak, and T_bk
        tree_dict[k].append(split_Trees(T_k)[0])
        tree_dict[k].append(split_Trees(T_k)[1])
    
    return tree_dict

"""
initializes all the X(x, T_0, c(x)), C is 0 indexed, however X_dict is 1 indexed
"""
def initialize_X(C, n):
    X_dict = {}

    for i in range(1, n+1):

        zero_key = tuple([0])
        color_key = tuple([C[i-1]])

        X_dict.setdefault(i, {}).setdefault(zero_key, {})[color_key] = 1
    
    return X_dict
        
"""
Returns the combinations of Color set C
"""
def findsubsets(C, n):
    return set(itertools.combinations(C, n))

"""
X function
"""
def X_func(X_dict, tree_dict, M, C, n, K, q):


    for k in range(K, 0, -1):
        
        T_k = tuple(tree_dict[k][0])
        d = tree_dict[k][1]
        T_a = tuple(tree_dict[k][2])
        T_b = tuple(tree_dict[k][3])


        colorSubsets = findsubsets(C, len(T_k))
        
        for x in range(1, n+1):
            for Cs in colorSubsets:
                #resultingSum is X(x, T_k, C)
                resultingSum = 0


                outerSum = 0
                for y in range(1, n+1):

                    if y == x:
                        continue

                    if(M[x-1][y-1] == 0):
                        continue

                    innerSum = 0
    
                    #dividing C into C1 and C2 wher C1 are the colors in Tak and C2 are the colors in Tbk
                    c1c2Subset = set(findsubsets(Cs, len(T_b)))
                    # print("C1c2 subset:", c1c2Subset)
                    for i in c1c2Subset:
                        
                        #tupple subtraction
                        c2 = set(i)
                        c1 = set(Cs) - c2

                        # print(Cs, c1, c2)

                        if len(c1) < len(T_a) or len(c2) < len(T_b):
                            # print("SKIPPED")
                            continue

                        c2_key = i
                        c1_key = tuple(c1)
                        
                        try:
                            innerSum += (X_dict[x][T_a][c1_key] * X_dict[y][T_b][c2_key] * M[x-1][y-1])
                        except KeyError:
                            continue
                    
                    outerSum += innerSum
                
                resultingSum = outerSum / d
                Cs_key = tuple(sorted(Cs))
                X_dict.setdefault(x, {}).setdefault(T_k, {})[Cs_key] = resultingSum
    
    # for keys, values in X_dict.items():
    #      print(keys, values)

    finalSum = 0
    finalTreeKey = tuple(tree_dict[1][0])
    finalColorKey = tuple(list(range(1,K+2)))
    for x in range(1, n+1):
        try:
            finalSum += X_dict[x][finalTreeKey][finalColorKey]
        except:
            continue
    
    return finalSum / q

"""
This method returns q
"""
def check_equality(tree):
    if tree.count(1) < 2:
        return 2 if len(tree) == 2 else 1
    m = tree.index(1,2)
    L1 = [tree[i] - 1 for i in range(1,m)]
    L2 = [tree[i] for i in range(m,len(tree))]
    L2.insert(0,0)
    if L1 == L2:
        return 2
    return 1


def algorithmOne(tree, M, C):
    t0 = time.time()
    edges = get_edges(tree)

    K = len(edges)
    n = len(M)


    finalSum = 0


    # print(C)

    X_dict = initialize_X(C, n)

    # for keys, values in X_dict.items():
    #      print(keys, values)


    tree_dict = get_Trees(tree, edges)

    # for keys, values in tree_dict.items():
    #      print(keys, values)

    q = check_equality(tree)

    # print("q", q)
    
    finalSum = X_func(X_dict, tree_dict, M, C, n, K, q)

    t1 = time.time()
    # print("Time:", t1-t0)
    return finalSum

def getX(i):
    global freeTrees
    global A
    global B
    sumX = 0
    n = len(A)
    CA = rand_assign(K, n)
    CB = rand_assign(K, n)
    for keys, values in freeTrees.items():
        sumX += (values * algorithmOne(keys, A, CA) * algorithmOne(keys, B, CB))
    return sumX

if __name__ == '__main__':
    start = time.time()

    K = 4

    freeTrees = generateFreeTrees(K+1)

    # tree = [0, 1, 2, 3, 3, 3, 2, 1, 2, 2, 1, 2, 2]
    A = [[0, 1, 1, 0, 1, 1],
         [1, 0, 1, 0, 0, 0],
         [1, 1, 0, 1, 1, 0],
         [0, 0, 1, 0, 1, 0],
         [1, 0, 1, 1, 0, 0],
         [1, 0, 0, 0, 1, 0]]
    
    B = [[0, 1, 0, 0, 1, 0],
         [1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 0],
         [0, 0, 1, 0, 1, 1],
         [1, 1, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0]]

    r = math.factorial(K+1) / math.pow(K+1, K+1)

    t = int(math.ceil(1/(math.pow(r,2))))

    pool = mp.Pool(mp.cpu_count())
    results = pool.map_async(getX, [i for i in range(t)]).get()
    pool.close()
    sumY = sum(results) / t
    print(sumY)

    print(time.time() - start)