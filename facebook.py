"""
Created on 6/28/21

@author: fayfayning
"""

import numpy as np
import pandas as pd
import math
import random
from simulations import alg2_fetch, erd_ren, calculateExpectedValueOne, \
    calculateExpectedValueTwo
from tree_generation import generateFreeTrees, aut
from get_x import get_edges
from scipy.sparse import csr_matrix
from scipy import io
import os
from os import listdir
from itertools import combinations_with_replacement
import re

def sample_n(A, B, n, p):
    x = random.randint(0, len(A) - n - 1)
    A = A[x:x + n, x:x + n]
    B = B[x:x + n, x:x + n]
    for i in range(len(A)):
        for j in range(i):
            temp1 = random.random()
            temp2 = random.random()
            if A[i,j] == 1:
                if temp1 < p:
                    A[i,j] = 1 - p
                    A[j,i] = A[i,j]
                else:
                    A[i,j] = 0 - p
                    A[j,i] = A[i,j]
            if B[i,j] == 1:
                if temp2 < p:
                    B[i,j] = 1 - p
                    B[j,i] = B[i,j]
                else: 
                    B[i,j] = 0 - p
                    B[j,i] = B[i,j]

    return [A, B]


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
                if matrix[i, j] == 1:
                    edge_list[1] += 1
                    string = "\n" + " ".join([str(i), str(j)])
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


def subsample_run(A, K, n):
    samples = sample_n(A, A, n)
    T = generateFreeTrees(K)
    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    t = int(math.ceil(1 / (math.pow(r, 2))))
    return alg2_fetch(T, samples[0], samples[1], K, t)


def run_two_networks(A, B, K, n):
    samples = sample_n(A, B, n)
    T = generateFreeTrees(K)
    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    t = int(math.ceil(1 / (math.pow(r, 2))))
    return alg2_fetch(T, samples[0], samples[1], K, t)


def get_expected_list(top, increment, p, K, ):
    ret = []
    for x in range(50, top, increment):
        ret.append(calculateExpectedValueOne(1, x, p, K))
    return ret

def get_sim1_txt(n, p, file):
    graph = erd_ren(n, p)
    to_edge_list(graph, file)

def convert_all_files(directory):
    save_path = 'all_out'
    for filename in os.listdir(directory):
        print(filename)
        new_name = filename[0:-4] + '.txt'
        completeName = os.path.join(save_path, new_name)
        to_edge_list(read_mat('facebook100/' + filename),completeName)


def expected_sim1():
    K = 6

    t = get_t(K)
    print('t =', t)

    H = [0, 1, 2, 2, 1, 2, 2]
    p = 0.001

    lst = [i * 50 for i in range(1, 21)]
    exp_val = []
    for n in lst:
        exp = calculateExpectedValueOne(1, n, p, H)
        exp_val.append(exp)
        name = "graphs/{}.txt".format(str(n))
        # to_edge_list(erd_ren(n, p), name)
    print(exp_val)


def get_t(K):
    r = math.factorial(K + 1) / math.pow(K + 1, K + 1)
    t = int(math.ceil(1 / (math.pow(r, 2))))
    return t

def edge_list_many():
    in_direct = "facebook100/"
    out_direct = "out_fb/"
    list = ["Simmons81", "USFCA72"]
    for i in list:
        to_edge_list(read_mat(in_direct + i + ".mat"), out_direct + i + ".txt")

def output_to_df(filename):
    df = pd.read_csv(filename, names=["File A", "File B", "n",
                                                   "Similarity"])

    schools = []
    school_dict = {}
    for i in df["File A"]:
        if i not in schools:
            schools.append(i)
            school_dict[i] = pd.DataFrame()

    self_df = {}

    for index, row in df.iterrows():
        if row["File A"] == row["File B"]:
            tup = tuple([row['File A'], row['n']])
            self_df[tup] = row['Similarity']

    for index, row in df.iterrows():
        if row["File A"] != row["File B"]:
            tup_1 = tuple([row['File A'], row['n']])
            tup_2 = tuple([row['File B'], row['n']])
            new_row = row
            new_row.loc['Similarity'] = row.loc['Similarity'] / math.sqrt(
                self_df[tup_1]) / math.sqrt(self_df[tup_2])
            school_dict[row["File A"]] = school_dict[row["File A"]].append(
                new_row)
            school_dict[row["File B"]] = school_dict[row["File B"]].append(
                new_row)

    for i in schools:
        print(i)
        print(school_dict[i])

    print(df)
    print(self_df)

def get_file_list(directory):
    for file in listdir(directory):
        if file != "gitkeep.txt" and ".txt" in file:
            print(file)

def one_per_array(directory):
    int = 1
    for file in listdir(directory):
        if file != "gitkeep.txt" and ".txt" in file:
            filename = "file_lists/{}_lst.txt".format(int)
            write_file = open(filename, "w")
            write_file.write(file)
            write_file.close()
            int += 1

def simi_spec_combo(directory, dict, combo):
    path1 = "{}/{}".format(directory, combo[0])
    file1 = open(path1, 'r')
    read_1 = file1.readlines()
    file1.close()

    path2 = "{}/{}".format(directory, combo[1])
    file2 = open(path2, 'r')
    read_2 = file2.readlines()
    file2.close()

    lines1 = len(read_1)
    lines2 = len(read_2)
    line_num = min(lines1, lines2)

    df_part = pd.DataFrame()

    for i in range(1, line_num):
        line1 = read_1[i].split(',')
        line2 = read_2[i].split(',')
        vals = [float(line1[j]) * float(line2[j]) for j in range(len(line1))]
        val = sum(vals)
        new_row = {'File A': dict[combo[0]], 'File B': dict[combo[1]],
                   'n': int(i + 2), 'Similarity': val}
        df_part = df_part.append(new_row, ignore_index = True)
    return df_part


def counts_to_similarity(directory):
    df = pd.DataFrame()
    dict = {}
    file_lst = []
    for file in listdir(directory):
        if file != "gitkeep.txt" and ".txt" in file:
            file_lst.append(file)
            dict[file] = re.split('[_.]',file)[1]
    combos = combinations_with_replacement(file_lst, 2)
    #print(dict)
    #print(list(combos))
    for i in combos:
        #print(i)
        combo_data = simi_spec_combo(directory, dict, i)
        df = df.append(combo_data)
    print(df)
    #df.to_csv('Similarity_Unnormalized.txt', encoding='utf-8', index=False)
    return df

def normalize_df(df):
    schools = []
    school_dict = {}
    for i in df["File A"]:
        if i not in schools:
            schools.append(i)
            school_dict[i] = pd.DataFrame()

    self_df = {}

    for index, row in df.iterrows():
        if row["File A"] == row["File B"]:
            tup = tuple([row['File A'], row['n']])
            self_df[tup] = row['Similarity']
    print(self_df)

    new_df = pd.DataFrame()
    for index, row in df.iterrows():
        if row["File A"] != row["File B"]:
            tup_1 = tuple([row['File A'], row['n']])
            tup_2 = tuple([row['File B'], row['n']])
            new_row = row
            new_row.loc['Similarity'] = row.loc['Similarity'] /  math.sqrt(
                self_df[tup_1]) / math.sqrt(self_df[tup_2])
            school_dict[row["File A"]] = school_dict[row["File A"]].append(
                new_row)
            school_dict[row["File B"]] = school_dict[row["File B"]].append(
                new_row)
            new_df = new_df.append(new_row)

    for i in schools:
        #print(i)
        #print(school_dict[i])
        j = 0

    #new_df.to_csv('Similarity_Normalized.txt', encoding='utf-8', index=False)
    print(new_df)
    return new_df

def analyze_err_time(directory):
    os.chdir(directory)
    dict_timing = {}
    dict_num = {}
    dict_max = {}
    dict_err = {}
    unknown = []
    for file in listdir(directory):
        if '.out' in file:
            f = open(file, 'r')
            lines = f.readlines()
            f.close()
            name = lines[0].split(', ')[1]
            index1 = name.rfind('/')
            index2 = name.rfind('.')
            name = name[index1 + 1: index2]
            timings = [float(line.split(', ')[-1]) for line in lines]
            dict_timing[name] = timings
            dict_max[name] = len(timings) + 2

            ind1 = file.rfind('_')
            ind2 = file.rfind('.')
            num = int(file[ind1 + 1: ind2])
            dict_num[num] = name

            out_of_mem = 'Some of your processes may have been killed by the cgroup out-of-memory handler'

            err = file[:-3] + 'err'
            g = open(err, 'r')
            err_lines = g.readlines()
            if len(err_lines) == 0:
                dict_err[name] = 'No Error'
            elif out_of_mem in err_lines[0]:
                dict_err[name] = 'Out of Memory'
            else:
                dict_err[name] = 'Unknown'
                unknown.append(tuple([name, num]))

    print(dict_timing)
    print(dict_max)
    print(dict_num)
    print(dict_err)
    print(unknown)

    return [dict_timing, dict_max, dict_num, dict_err, unknown]

def comp_err_out():
    slurm_path_no = '/Users/fayfayning/Desktop/big_boi/slurms2/no'
    [no_timing, no_max, no_num, no_err, no_unk] = analyze_err_time(
        slurm_path_no)

    slurm_path_o = '/Users/fayfayning/Desktop/big_boi/slurms2/o'
    [o_timing, o_max, o_num, o_err, o_unk] = analyze_err_time(slurm_path_o)

    for key in o_timing.keys():
        print(key, 'no', no_timing[key])
        print(key, 'o', o_timing[key])
        print(key, 'no_err', no_err[key], no_max[key])
        print(key, 'o_err', o_err[key], o_max[key])

if __name__ == '__main__':
    small_fb_path = "/Users/fayfayning/Documents/GitHub/fasciaGraphSimilarity/small_fb"
    #get_file_list(small_fb_path)

    #one_per_array(small_fb_path)

    #output_to_df("small_fb_full_n.txt")

    count_tests_path = "/Users/fayfayning/Desktop/count_trees_no"
    df_og = counts_to_similarity(count_tests_path)
    normalize_df(df_og)

    #comp_err_out()