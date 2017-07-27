#!/usr/bin/env python

"""
Titanic Analysis
Exploratory plots are done in our kaggle notebook, so that won't be reproduced here.
This script will load the data, clean/preprocess it, then train a random forest and generate test predictions
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
    

def cleanData(data,lookup=None):
    """ 
    load the train and test data. 
    Any preprocessing derived from the train data is applied to the test data
    Create lookup table to impute age based on pclass and gender
    """
    #data = data.copy(deep=True)
    train = False
    if lookup is None:
        lookup = np.zeros([2,3])
        train = True
    sexdict = {0:'male', 1:'female'}
    

    # preprocessing/cleaning
    

    # map 'Sex' (string) to 'gender' (int)
    data['Sex'] = data['Sex'].astype('category')
    data['Embarked'] = data['Embarked'].astype('category')

    # use the training data to create age lookup table
    if train:
        for s in [0,1]:
            thesex = sexdict[s]
            for pc in range(0,3):
                lookup[s,pc] = data[ (data.Sex == thesex) & (data.Pclass == pc +1) ]['Age'].median()
    
    # fill dataframe based on lookup table
    for s in [0,1]:
        thesex = sexdict[s]
        for pc in range(0,3):
            data.loc[(data.Sex ==thesex ) & ( data.Pclass == pc +1) & (data.Age.isnull()),'Age'] = lookup[s,pc]
        
        # sum siblings spouses and parents
        #data['Family'] = data['SibSp'] + data['Parch']

    # code categorical varibales as one hot
    data = pd.get_dummies(data=data,columns=['Sex','Embarked','Pclass'])
    
    # drop some columns
    data = data.drop(['PassengerId','Ticket','Cabin','Name'],axis=1)


    # split training into train and validation set

    # Y = train_raw['Survived']
    # X = train_raw.drop('Survived',axis=1)

    # X_test = test_raw

    # X_train, X_val,Y_train,Y_val = \
    #     train_test_split(X,Y,test_size=val_frac,random_state=rstate)
    return data,lookup


def dostuff(splitseed=96,splitfrac=0.2):

    # load data
    train_raw = pd.read_csv('./data/titanic_train.csv')
    test_raw = pd.read_csv('./data/titanic_test.csv')

    # copy passengerid for test data
    # this is dropped in the cleaning (so our model is more than just a pid->survived lookup table)

    print(train_raw.describe())
    test_pid = test_raw['PassengerId'].copy(deep=True)

    train_clean, lookup = cleanData(train_raw)
    X_test ,lookup = cleanData(test_raw,lookup=lookup)

    Y = train_clean['Survived']
    X = train_clean.drop('Survived',axis=1)

    # split train into training and validation
    X_train, X_val, Y_train, Y_val = \
        train_test_split(X,Y,test_size=splitfrac,random_state=splitseed)

    print(X_train.describe())
    print(X_val.describe())


def main():
    dostuff()



if __name__ == '__main__':
    main()
