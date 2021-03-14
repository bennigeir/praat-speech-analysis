# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB


warnings.filterwarnings("ignore")


def is_outlier(s):
    
    lower_limit = s.mean() - (s.std() * 3)
    upper_limit = s.mean() + (s.std() * 3)
    
    return ~s.between(lower_limit, upper_limit)


def read_csv():
    
    # read csv file
    df = pd.read_csv('icelandic_vowels_f1_f2.csv')
    
    return df


def prepare_df(df):
    
    out = df.copy()
    
    # remove word kýr
    out = out[out.word != 'kýr']

    # create boolean column for native icelandic speaker
    out['native_isl'] = np.where(out['native_lang']=='isl', 'isl', 'not_isl')
    
    # remove outliers
    out = out[~out.groupby(by='word')['F1'].apply(is_outlier)]
    out = out[~out.groupby(by='word')['F2'].apply(is_outlier)]
    
    # reset index
    out.reset_index(inplace=True)
    
    return out


def get_x_y(df, name=False):
    
    # set Y
    y = df['native_isl']
    
    # set X
    # one-hot encode the words
    if not name:
        X = pd.get_dummies(df[['word']]).join(df)
        X = X.drop(columns=['name', 'age', 'native_lang', 'spoken_lang',
                            'word', 'native_isl'])
    else: 
        X = df.drop(columns=['name', 'age', 'native_lang', 'spoken_lang',
                             'word', 'native_isl'])
    
    return X, y


def scale_data(low, high, df):
    
    out = df.copy()
    
    scaler = MinMaxScaler(feature_range=(low, high))
    out = scaler.fit_transform(out)
    
    return out


def augment_data(df, augment_df):
    
    for i in range(-10,10):
        augmentation = df.copy(deep=True)
        augmentation[['F1', 'F2']] += i
        augment_df = pd.concat([augment_df, augmentation])

    augmented_X_train = augment_df.drop(columns=['native_isl'])
    augmented_y_train = augment_df[['native_isl']].values.ravel()
    
    return augmented_X_train, augmented_y_train   


def train_kf(X, y):
    
    scores, scores2, scores3 = [], [], []
    augmented_scores, augmented_scores2, augmented_scores3 = [], [], []
    ensemble_scores=[]
    ensemble_augmented_scores=[]
    
    kf = KFold(n_splits=2, random_state=10, shuffle=True)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        
        # split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
    
        # augment data (very simple augmentation method)
        training = X_train.join(Y_train.to_frame())
        augment_training = training.copy(deep=True)
        augmented_X_train, augmented_y_train = augment_data(training,
                                                            augment_training)
    
        # scale data
        X_train = scale_data(0, 1, X_train)
        augmented_X_train = scale_data(0, 1, augmented_X_train)
        X_test = scale_data(0, 1, X_test)
    
        # run and score models
        clf_fitted = clf.fit(X_train, Y_train)
        scores.append(clf_fitted.score(X_test, Y_test))
        clf_fitted = clf.fit(augmented_X_train, augmented_y_train)
        augmented_scores.append(clf_fitted.score(X_test, Y_test))
        
        clf2_fitted = clf2.fit(X_train, Y_train)
        scores2.append(clf2_fitted.score(X_test, Y_test))
        clf2_fitted = clf2.fit(augmented_X_train, augmented_y_train)
        augmented_scores2.append(clf2_fitted.score(X_test, Y_test))
        
        clf3_fitted = clf3.fit(X_train, Y_train)
        scores3.append(clf3_fitted.score(X_test, Y_test))
        clf3_fitted = clf3.fit(augmented_X_train, augmented_y_train)
        augmented_scores3.append(clf3_fitted.score(X_test, Y_test))
        
        preds = clf3.predict(X_test)
        
        # ensemble addition
        ensemble_fitted = ensemble.fit(X_train, Y_train)
        ensemble_scores.append(ensemble_fitted.score(X_test, Y_test)) 
        ensemble_fitted = ensemble.fit(augmented_X_train, augmented_y_train)
        ensemble_augmented_scores.append(ensemble_fitted.score(X_test, Y_test))
        
        word_scores = {}
        for x, pred, actual in zip(X_test, preds, Y_test):
            word = X.columns[np.argmax(x)]
            if word in word_scores:
                word_scores[word]['total'] += 1
            else:
                word_scores[word] = {}
                word_scores[word]['total'] = 1
                word_scores[word]['correct'] = 0
            if pred == actual:
                word_scores[word]['correct'] += 1
    
        for word in word_scores:
            print(word, word_scores[word]['correct'] / word_scores[word]['total'])
            
    print('SVC scores:', scores)
    print('SVC scores augmented:', augmented_scores)
    print('MLPClassifier scores:', scores2)
    print('MLPClassifier scores augmented:', augmented_scores2)
    print('KNeighboursClassifier scores: ', scores3)
    print('KNeighboursClassifier scores augmented: ', augmented_scores3)
    
    # ensemble addition
    print('Ensemble scores:', ensemble_scores)
    print('Ensemble scores augmented:', ensemble_augmented_scores)


def train_name(df):
        
    for name in set(df['name']):

        # set Y
        training = df[df.name!=name]
        test = df[df.name==name]    
        
        # set Y
        X_train, Y_train = get_x_y(training, name=True)
        X_test, Y_test = get_x_y(test, name=True)
            
        # augment data (very simple augmentation method)
        training = X_train.join(Y_train.to_frame())
        augment_training = training.copy(deep=True)
        augmented_X_train, augmented_y_train = augment_data(training,
                                                            augment_training)
    
        # scale data
        X_train = scale_data(0, 1, X_train)
        augmented_X_train = scale_data(0, 1, augmented_X_train)
        X_test = scale_data(0, 1, X_test)
    
        # run and score models
        # clf_fitted = clf.fit(X_train, Y_train)
        # scores.append(clf_fitted.score(X_test, Y_test))
        # clf_fitted = clf.fit(augmented_X_train, augmented_y_train)
        # augmented_scores.append(clf_fitted.score(X_test, Y_test))
        
        # clf2_fitted = clf2.fit(X_train, Y_train)
        # scores2.append(clf2_fitted.score(X_test, Y_test))
        # clf2_fitted = clf2.fit(augmented_X_train, augmented_y_train)
        # augmented_scores2.append(clf2_fitted.score(X_test, Y_test))
        
        clf3_fitted = clf3.fit(X_train, Y_train)
        scores3 = (clf3_fitted.score(X_test, Y_test))
        clf3_fitted = clf3.fit(augmented_X_train, augmented_y_train)
        augmented_scores3 = (clf3_fitted.score(X_test, Y_test))
        
        preds = clf3.predict(X_test)
    
        lr_score = lr.fit(X_train, Y_train).score(X_test, Y_test)
        rfc_score = rfc.fit(X_train, Y_train).score(X_test, Y_test)
        dtc_score = dtc.fit(X_train, Y_train).score(X_test, Y_test)
        mlp_score = mlp.fit(X_train, Y_train).score(X_test, Y_test)
        
        ensemble_fitted = ensemble.fit(X_train, Y_train)
        ensemble_scores = (ensemble_fitted.score(X_test, Y_test)) 
        ensemble_fitted = ensemble.fit(augmented_X_train, augmented_y_train)
        ensemble_augmented_scores = (ensemble_fitted.score(X_test, Y_test))

        # word_scores = {}
        # for x, pred, actual in zip(X_test, preds, Y_test):
            # word = X.columns[np.argmax(x)]
            # if word in word_scores:
                # word_scores[word]['total'] += 1
            # else:
                # word_scores[word] = {}
                # word_scores[word]['total'] = 1
                # word_scores[word]['correct'] = 0
            # if pred == actual:
                # word_scores[word]['correct'] += 1
    
        # for word in word_scores:
            # print(word, word_scores[word]['correct'] / word_scores[word]['total'])
    
        # print('SVC scores:', scores)
        # print('SVC scores augmented:', augmented_scores)
        # print('MLPClassifier scores:', scores2)
        # print('MLPClassifier scores augmented:', augmented_scores2)
        
        print('\n' + 20*'*' + '\n' + name + '\n' + 20*'*')      
        
        print('KNeighboursClassifier scores: ', scores3)
        print('KNeighboursClassifier scores augmented: ', augmented_scores3)
        
        print('LogisticRegression scores: ', lr_score)
        # print('LogisticRegression scores augmented: ', augmented_scores3)
        
        print('RandomForestClassifier scores: ', rfc_score)
        # print('RandomForestClassifier scores augmented: ', augmented_scores3)
        
        print('DecisionTreeClassifier scores: ', dtc_score)
        # print('DecisionTreeClassifier scores augmented: ', augmented_scores3)
        
        print('MLPClassifier scores: ', mlp_score)
        # print('MLPClassifier scores augmented: ', augmented_scores3)
        
        
        # ensemble addition
        print('Ensemble scores:', ensemble_scores)
        print('Ensemble scores augmented:', ensemble_augmented_scores)


#######################################

df = read_csv()

prep_df = prepare_df(df)
X, y = get_x_y(prep_df)

# define classification model
clf = svm.SVC()
clf2 = MLPClassifier(solver='lbfgs',
                     alpha=1e-5,
                     hidden_layer_sizes=(5, 2),
                     random_state=1, max_iter=200)
clf3 = KNeighborsClassifier()

#######################################

# ensemble
estimator = []
estimator.append(('MLP', MLPClassifier(solver='lbfgs',
                                       alpha=1e-5,
                                       hidden_layer_sizes=(5, 2),
                                       random_state=1,
                                       max_iter=200))) 
estimator.append(('DTC', DecisionTreeClassifier())) 
estimator.append(('MNB', MultinomialNB())) 
ensemble = VotingClassifier(estimators=estimator)

# train_kf(X, y)

#######################################

# ensemble addition
lr = LogisticRegression()
rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
knc = KNeighborsClassifier()
mlp = MLPClassifier()

estimator = []
estimator.append(('LR', lr)) 
estimator.append(('RFC', rfc)) 
estimator.append(('DTC', dtc)) 
estimator.append(('KNC', knc)) 
estimator.append(('MLP', mlp)) 
ensemble = VotingClassifier(estimators=estimator, weights=[1,1,2,3,1])

train_name(prep_df)