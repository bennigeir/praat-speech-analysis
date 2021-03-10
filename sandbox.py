# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB


def is_outlier(s):
    
    lower_limit = s.mean() - (s.std() * 3)
    upper_limit = s.mean() + (s.std() * 3)
    
    return ~s.between(lower_limit, upper_limit)


# read csv file
df = pd.read_csv('icelandic_vowels_f1_f2.csv')

# remove word kýr
df = df[df.word != 'kýr']

# create boolean column for native icelandic speaker
df['native_isl'] = np.where(df['native_lang']=='isl', 'isl', 'not_isl')

# remove outliers
df = df[~df.groupby(by='word')['F1'].apply(is_outlier)]
df = df[~df.groupby(by='word')['F2'].apply(is_outlier)]

# reset index
df.reset_index(inplace=True)

# set Y
Y = df['native_isl']

# set X
    # one-hot encode the words
X = pd.get_dummies(df[['word']]).join(df)

# define classification model
clf = svm.SVC()
clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=200)
clf3 = KNeighborsClassifier()

# ensemble addition
estimator = []
estimator.append(('MLP', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=200))) 
estimator.append(('DTC', DecisionTreeClassifier())) 
estimator.append(('MNB', MultinomialNB())) 
ensemble = VotingClassifier(estimators=estimator)

# define data splitting method
kf = KFold(n_splits=2, random_state=10, shuffle=True)
kf.get_n_splits(X)
print(kf)

scores = []
augmented_scores = []
scores2 = []
augmented_scores2 = []
scores3 = []
augmented_scores3 = []

ensemble_scores=[]
ensemble_augmented_scores=[]

for train_index, test_index in kf.split(X):
    # split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    X_train = X_train.drop(columns=['name', 'age', 'native_lang', 'spoken_lang', 'word', 'native_isl'])
    X_test = X_test.drop(columns=['name', 'age', 'native_lang', 'spoken_lang', 'word', 'native_isl'])

    # augment data (very simple augmentation method)
    training = X_train.join(Y_train.to_frame())
    augment_training = training.copy(deep=True)
    for i in range(-10,10):
        augmentation = training.copy(deep=True)
        augmentation[['F1', 'F2']] += i
        augment_training = pd.concat([augment_training, augmentation])

    augmented_X_train = augment_training.drop(columns=['native_isl'])
    augmented_Y_train = augment_training[['native_isl']].values.ravel()

    # scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    augmented_X_train = scaler.fit_transform(augmented_X_train)
    X_test = scaler.fit_transform(X_test)

    # # run model
    scores.append(clf.fit(X_train, Y_train).score(X_test, Y_test))
    augmented_scores.append(clf.fit(augmented_X_train, augmented_Y_train).score(X_test, Y_test))
    scores2.append(clf2.fit(X_train, Y_train).score(X_test, Y_test))
    augmented_scores2.append(clf2.fit(augmented_X_train, augmented_Y_train).score(X_test, Y_test))
    scores3.append(clf3.fit(X_train, Y_train).score(X_test, Y_test))
    augmented_scores3.append(clf3.fit(augmented_X_train, augmented_Y_train).score(X_test, Y_test))
    preds = clf3.predict(X_test)
    
    # ensemble addition
    ensemble_scores.append(ensemble.fit(X_train, Y_train).score(X_test, Y_test))
    ensemble_augmented_scores.append(ensemble.fit(augmented_X_train, augmented_Y_train).score(X_test, Y_test))
    
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