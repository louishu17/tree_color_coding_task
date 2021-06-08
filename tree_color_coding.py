"""
Created on 6/7/21

@author: louishu17, fayfayning, kieranlele
"""

"""task one (I think this should work/we need it)"""
def check_equality(tree):
    if tree.count(1) < 2:
        return False
    m = tree.index(1,2)
    L1 = [tree[i] - 1 for i in range(1,m)]
    L2 = [tree[i] for i in range(m,len(tree))]
    L2.insert(0,0)
    if L1 == L2:
        print("L1 and L2 are the same")
        return 2
    return 1

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


def get_Trees(tree_level, edges):

    #Edges list is indexed from 0, so edges[0] = edge 1

    tree_dict = {}

    tree_dict[0] = [0]

    for k in range(len(edges), 0, -1):
        tree_dict[k] = []
        r_double_prime = len(tree_level) 

        #R is edges[k][1]-1
        for j in range(edges[k-1][0], len(edges)+1):

            if tree_level[j] == tree_level[edges[k-1][0]-1]:
                r_double_prime = j
                break

        #from l_R+1 to l_R'' - l_R'
        tree_dict[k].append(0)
        for j in range(edges[k-1][1], r_double_prime+1):
            tree_dict[k].append(tree_level[j-1] - tree_level[edges[k-1][0]-1])
            
        print()
    
    return tree_dict


def split_Trees(trees):
    for keys, values in trees.items():
        if(len(values) == 1):
            continue

        first_1_ind = values.index(1)
        second_exists = True
        try:
            second_1_ind = values.index(1, first_1_ind+1)
        except ValueError:
            second_exists = False
        
        if(second_exists):
            Tbk = [values[i] - 1 for i in range(1,second_1_ind)]
            Tak = [values[i] for i in range(second_1_ind,len(values))]
            Tak.insert(0,0)

            print(keys, Tbk, Tak)
        
        else:
            Tbk = [values[i]-1 for i in range(1, len(values))]
            Tak = [0]

            print(keys, Tbk, Tak)


def get_overcounting(tree):

    #USE DICTIONARY


    pieceset = set()

    first_1_ind = tree.index(1)
    second_1_ind = tree.index(1, first_1_ind+1)
     
    tup = tuple(tree[first_1_ind:second_1_ind])
    print(tup)
    pieceset.add(tup)

    counter = 1

    left = 0
    right = 0

    for i in range(second_1_ind, len(tree)):
        if(tree[i] == 1 and tree[left] != 1):
            left = i
        elif(tree[i] == 1 and tree[left] == 1):
            right = i
        
        if(tree[left] == 1 and tree[right] == 1):
            piece = tuple(tree[left:right])
            print(piece)

            if piece in pieceset:
                counter += 1

    return counter

if __name__ == '__main__':
    tree = [0, 1, 2, 2, 1, 2, 2, 1]

    edges = get_edges(tree)

    trees = get_Trees(tree, edges)

    split_Trees(trees)

    count = get_overcounting(tree)

    print(count)
