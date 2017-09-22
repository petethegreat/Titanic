#!/usr/bin/env python

"""
Titanic Analysis
Exploratory plots are done in our kaggle notebook, so that won't be reproduced here.
This script will load the data, clean/preprocess it, then train a random forest and generate test predictions
"""
from __future__ import print_function
import random
import re
import numpy as np
import pandas as pd

# from matplotlib import pyplot as plt
# import matplotlib
# import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
LOOKUP = {}
MAJORITY = 1

# Wrapper for some model tuning
class ModelStuff(object):
    '''
    Wrapper class to hold models and their tuning parameters
    '''
    def __init__(self, name, model, cvargs):
        self.Name = name
        self.Model = model
        self.CVargs = cvargs
        self.TunedModel = None
        self.CVgrid = None
    ##############################################################

    def tuneModel(self, xtrain, ytrain):
        print('Tuning {name}'.format(name=self.Name))
        cvgrid = GridSearchCV(self.Model, **self.CVargs)
        cvgrid.fit(xtrain, ytrain)
        #oob_score
        # print('best rf oob_score = {oob}'.format(oob=cvgrid.best_estimator_.oob_score_))
        print('{name:20} best cv score = {cvs}'.format(name=self.Name, cvs=cvgrid.best_score_))
        self.CVgrid = cvgrid
        self.TunedModel = cvgrid.best_estimator_
##############################################################

def cleanData(data, train=True):
    """
    load the train and test data.
    Any preprocessing derived from the train data is applied to the test data
    Create lookup table to impute age based on pclass and gender
    """
    #data = data.copy(deep=True)
    # global LOOKUP
    if train:
        #median age by gender and pclass
        LOOKUP['AgeGenderPclass'] = np.zeros([2, 3])
        # median fare by pclass
        LOOKUP['FarePclass'] = np.zeros([3])
    sexdict = {0:'male', 1:'female'}

    # preprocessing/cleaning
    # map 'Sex' (string) to 'gender' (int)
    data['Sex'] = data['Sex'].astype('category')
    data['Embarked'] = data['Embarked'].astype('category')
    data['Family'] = data['SibSp'] + data['Parch']

    # data['Title'] = data['Name'].map(lambda x: x.split()[0])
    titleregex = r'^(?P<surname>[^, ]+), (?P<Title>[^\s]+)'
    NameStuff = data['Name'].str.extract(titleregex, expand=True)
    data['Title'] = NameStuff['Title']
    data['Title'] = data['Title'].astype('category')

    # use the training data to create age and fare lookup tables, to deal with missing data
    if train:
        for pc in range(0, 3):
            #median fare per pclass
            LOOKUP['FarePclass'][pc] = data[data.Pclass == pc+1]['Fare'].median()
            for s in [0, 1]:
                # median age by pclass and gender
                thesex = sexdict[s]
                LOOKUP['AgeGenderPclass'][s, pc] = data[
                    (data.Sex == thesex) &
                    (data.Pclass == pc +1)
                    ]['Age'].median()

    # fill missing values in dataframe based on lookup table
    for pc in range(0, 3):
        data.loc[(data.Fare.isnull()) & (data.Pclass == pc+1), 'Fare'] = LOOKUP['FarePclass'][pc]

        for s in [0, 1]:
            thesex = sexdict[s]
            data.loc[
                (data.Sex == thesex) & (data.Pclass == pc +1) &
                (data.Age.isnull()), 'Age']  \
            = LOOKUP['AgeGenderPclass'][s, pc]

        # sum siblings spouses and parents
        #data['Family'] = data['SibSp'] + data['Parch']

    # code categorical varibales as one hot
    data = pd.get_dummies(data=data, columns=['Sex', 'Embarked', 'Pclass', 'Title'])
    if train:
        LOOKUP['columns_set'] = set(data.drop(['Survived'], axis=1).columns)
    else:
        missing_cols = LOOKUP['columns_set'] - set(data.columns)
        for c in missing_cols:
            data[c] = 0

        # test should be missing nothing now
        extra_cols = set(data.columns) - LOOKUP['columns_set']

        if len(extra_cols) > 0:
            print(
                'the following features are present in test but were absent'+
                ' in training and will be dropped:')
        for c in extra_cols:
            print('  -{c}'.format(c=c))
        data = data.drop(extra_cols, axis=1)

    # drop some columns
    data = data.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)
    return data


##############################################################

def printCVresults(thecvgrid):
    ''' A couple things in here are specific to random forests '''
    ranking = thecvgrid.cv_results_['rank_test_score']
    # list of Nones to hold output
    strs = [None for i in ranking ]

    #loop over grid results, insert performance in appropriate place in list
    for index, rank in enumerate(ranking):
        mean = thecvgrid.cv_results_['mean_test_score'][index]
        stddev = thecvgrid.cv_results_['std_test_score'][index]
        # format the results
        thestr = '{r:2}- max_features: {mf:10}, n_estimators: {ne:4}, {m:5.4g} +/- {sd:5.4g}'.format(
            r=rank,
            mf=thecvgrid.cv_results_['params'][index]['max_features'],
            ne=thecvgrid.cv_results_['params'][index]['n_estimators'],
            m=mean,
            sd=stddev
        )
        
        # if there are ties, adjust the ranking index
        # (if two parameter combinations score the same,
        # the second is ordered as if it were ranked one worse)
        checktie = rank -1 
        while strs[checktie]:
            checktie += 1
        strs[checktie] = thestr

    # print stuff
    for i in strs:
        print(i)
##############################################################

def setupModels():
    ''' Set up the models that will comprise our ensemble '''
    Models = []
    global MAJORITY

    # random Forest
    cvargs = {
        'param_grid':{
            'max_features': ['log2', 'sqrt', 3],
            'n_estimators': [100, 200, 300, 500]},
        'cv':10,
        'n_jobs':2,
        'refit':True,
    }
    Models.append(
        ModelStuff(
            name='Random Forest',
            model=RandomForestClassifier(oob_score=True),
            cvargs=cvargs)
        )

    # AdaBoost
    cvargs = {
        'param_grid':{
            'n_estimators':[100, 200, 300, 500]},
        'cv':10,
        'n_jobs':2,
        'refit':True,
    }
    Models.append(
        ModelStuff(
            name='AdaBoost',
            model=AdaBoostClassifier(),
            cvargs=cvargs)
        )

    # LDA
    cvargs = {
        'param_grid':{
            'solver':['svd', 'lsqr'],
            # 'shrinkage':['auto', 'none']
            },
        'cv':10,
        'n_jobs':2,
        'refit':True,
    }
    Models.append(
        ModelStuff(
            name='LDA',
            model=LinearDiscriminantAnalysis(),
            cvargs=cvargs)
        )


    ## SVM
    ## Seems to take a long time, try caching
    # cvargs = {
    #     'cv':3,
    #     'n_jobs':2,
    #     'refit':True,
    #     'param_grid':{
    #         'kernel':[ 'linear', 'poly' ]
    #         }
    #     }
    # Models.append(
    #     ModelStuff(
    #         name='SVM',
    #         model=SVC(random_state=37, cache_size=1024),
    #         cvargs=cvargs)
    #     )

    # Linear Discriminant Analysis
    MAJORITY = 2
    return Models

##############################################################

def dostuff(splitseed=96, splitfrac=0.2):
    '''
    Split the training data
    tune and evaluate models
    predict test set
    '''
    useval = splitfrac > 0.0

    # load data
    train_raw = pd.read_csv('./data/titanic_train.csv')
    test_raw = pd.read_csv('./data/titanic_test.csv')

    # copy passengerid for test data
    # this is dropped in the cleaning (so our model is more than just a pid->survived lookup table)
    test_pid = test_raw['PassengerId'].copy(deep=True)
    test_pid.columns = ['PassengerId']

    train_clean = cleanData(train_raw, train=True)
    X_test = cleanData(test_raw, train=False)

    Y = train_clean['Survived']
    X = train_clean.drop('Survived', axis=1)
    if useval:
        # split train into training and validation
        X_train, X_val, Y_train, Y_val = \
            train_test_split(X, Y, test_size=splitfrac, random_state=splitseed)
    else:
        X_train = X
        Y_train = Y

    # set the seed for reproducability
    random.seed(42)

    # Models
    Models = setupModels()

    for themodel in Models:
        themodel.tuneModel(X_train, Y_train)

    # score on the validation set, if appropriate
    if useval:
        for themodel in Models:
            val_score = themodel.TunedModel.score(X_val, Y_val)
            print('{mname} validation score = {vs:5.4g}'.format(mname=themodel.Name, vs=val_score))

    # make predictions
    surv_cols = []
    TestPredictions = pd.DataFrame({'PassengerId':test_pid})
    for themodel in Models:
        cname = 'surv_{mname}'.format(mname=re.sub(' ', '_', themodel.Name))
        surv_cols.append(cname)
        predicted = themodel.TunedModel.predict(X_test)
        TestPredictions[cname] = predicted

    # form an output dataframe, write to csv
    # test_out = pd.DataFrame({'PassengerId':test_pid, 'Survived':predicted})

    # need to count votes
    TestPredictions['Survived'] = np.where(TestPredictions[surv_cols].sum(axis=1) >= MAJORITY, 1, 0)
    TestPredictions = TestPredictions.set_index('PassengerId')
    print(TestPredictions.head())

    # test_out.to_csv('./data/pete_titanic_rf_predictions.csv')
    print(TestPredictions[['Survived']].head())
    TestPredictions[['Survived']].to_csv('./data/ensemble_test.csv')#.format(sf=splitfrac))
    # The ensemble scores 77% on Kaggle

##############################################################

def main():
    ''' call the function that does stuff '''
    dostuff(splitfrac=0.0)

##############################################################

if __name__ == '__main__':
    main()
