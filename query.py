import pandas as pd
import numpy as np


def which_state(colleges):
    num = 'yes'
    asked = []
    while (num == 'yes'):
        states = input('What state would you like to match with?')
        if (states in colleges.state.values):
            asked.append(states)
            num = input('Would you like to add more states?')
        else:
            print('We do not have any data on that state')
            num = input('Would you like to try again?')
    colleges = colleges[colleges.state.isin(asked)]
    return colleges

#not yet done
def which_size(colleges):
    num = 'yes'
    asked = []
    while (num == 'yes'):
        sizes = input('Would you like large or small schools?')
        if (sizes == 'large' or sizes == 'small'):
            asked.append(sizes)
            num = 'no'
        else:
            print('We do not have any data on schools that size')
            num = input('Would you like to try again?')
    colleges = colleges[colleges.size.isin(asked)]
    return colleges


def which_school(colleges, only_one):
    num = 'yes'
    asked = []
    while (num == 'yes'):
        school = input('What school would you like to match with?')
        if (school in colleges.school_name.values):
            asked.append(school)
            if (only_one):
                return colleges[colleges.school_name.isin(asked)]
            num = input('Would you like to add more schools?')
        else:
            print('We do not have any data on that state')
            num = input('Would you like to try again?')
    colleges = colleges[colleges.school_name.isin(asked)]
    return colleges







if __name__ == '__main__':
    colleges = pd.read_csv('college_database.csv')
    print(colleges.school_name)
    colleges.drop(colleges.columns[[0]], axis=1, inplace=True)
    goon = 'yes'
    school = which_school(colleges, True)
    while (goon == 'yes'):
        which = input("state, size, or school?")
        if (which == 'state'):
            colleges = which_state(colleges)
            print(colleges)
            if(len(colleges)) == 0:
                print('You have run out of colleges')
                break
            goon = input('Would you like to pick another category?')
        if (which == 'size'):
            colleges = which_size(colleges)
            if(len(colleges)) == 0:
                print('You have run out of colleges')
                break
            goon = input('Would you like to pick another category?')
            print(colleges)
        if (which == 'school'):
            colleges = which_school(colleges, False)
            if(len(colleges)) == 0:
                print('You have run out of colleges')
                break
            goon = input('Would you like to pick another category?')
            print(colleges)
    #pseudo code
    print(colleges)
    '''
    scores = []
    for x in (colleges)
    scpres.append((cpp_code(school.filename),x.school_name))
    '''