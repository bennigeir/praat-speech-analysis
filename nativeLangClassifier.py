# import libraries

import pandas as pd
import random
import numpy as np
from sklearn import svm
import scipy
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

def is_outlier(s):
    lower_limit = s.mean() - (s.std() * 3)
    upper_limit = s.mean() + (s.std() * 3)
    return ~s.between(lower_limit, upper_limit)

# read csv file
df = pd.read_csv('icelandic_vowels_f1_f2.csv')

# create boolean column for native icelandic speaker
df['native_isl'] = np.where(df['native_lang']=='isl', 'isl', 'not_isl')

# remove outliers
df = df[~df.groupby(by='word')['F1'].apply(is_outlier)]
df = df[~df.groupby(by='word')['F2'].apply(is_outlier)]

# reset index
df.reset_index(inplace=True)

# set X and Y
Y = df['native_isl'].values.ravel()
    # one-hot encode the words
X = pd.get_dummies(df[['word']]).join(df)
X = X.drop(columns=['name', 'age', 'native_lang', 'spoken_lang', 'word', 'native_isl'])

# define classification model
clf = svm.SVC()

# define classification model
clf = svm.SVC()
clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf3 = KNeighborsClassifier()

# define data splitting method
kf = KFold(n_splits=2, random_state=11, shuffle=True)
kf.get_n_splits(X)
print(kf)

scores = []
scores2 = []
scores3 = []
for train_index, test_index in kf.split(X):
    # split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # run model
    scores.append(clf.fit(X_train, Y_train).score(X_test, Y_test))
    scores2.append(clf2.fit(X_train, Y_train).score(X_test, Y_test))
    scores3.append(clf3.fit(X_train, Y_train).score(X_test, Y_test))

print('SVC scores: ', scores)
print('MLPClassifier scores:', scores2)
print('KNeighboursClassifier scores: ', scores3)
