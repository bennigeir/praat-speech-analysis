# -*- coding: utf-8 -*-

import warnings
import matplotlib.pyplot as plt

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

from sklearn.metrics import classification_report
from utils import (read_csv,
                   prepare_df,
                   get_x_y,
                   scale_data,
                   augment_data,
                   )

warnings.filterwarnings("ignore")


def train_vowel(df):

    work_df = df.copy()
    work_df = work_df.drop(columns=['native_isl'])
    work_df = work_df[work_df.native_lang == 'isl']

    # make training test splits:
    for name in set(work_df['name']):
    
        training = work_df[work_df.name!=name]
        test = work_df[work_df.name==name]
    
        # set Y
        Y_train = training['word']
        Y_test = test['word']
    
        # set X
        X_train = training.drop(columns=['name', 'age', 'native_lang',
                                         'spoken_lang', 'word'])
        X_test = test.drop(columns=['name', 'age', 'native_lang',
                                    'spoken_lang', 'word'])
    
    
        # augment data (very simple augmentation method)
        training = X_train.join(Y_train.to_frame())
        augment_training = training.copy(deep=True)
        augmented_X_train, augmented_y_train = augment_data(training,
                                                            augment_training,
                                                            'word')
    
        # scale data
        X_train = scale_data(0, 1, X_train)
        augmented_X_train = scale_data(0, 1, augmented_X_train)
        X_test = scale_data(0, 1, X_test)
    
        # # run model
        score = clf.fit(X_train, Y_train).score(X_test, Y_test)
        Y_pred = clf.predict(X_test)
        # a_score = a_clf.fit(augmented_X_train, augmented_Y_train).score(X_test, Y_test)
        # score2 = clf2.fit(X_train, Y_train).score(X_test, Y_test)
        # a_score2 = a_clf2.fit(augmented_X_train, augmented_Y_train).score(X_test, Y_test)
        # score3 = clf3.fit(X_train, Y_train).score(X_test, Y_test)
        # predicted3 = clf3.predict(X_test)
        # a_score3 = a_clf3.fit(augmented_X_train, augmented_Y_train).score(X_test, Y_test)
        # a_predicted3 = a_clf3.predict(X_test)
        # score4 = clf4.fit(X_train, Y_train).score(X_test, Y_test)
        # score5 = clf5.fit(X_train, Y_train).score(X_test, Y_test)
    
        print(name, score)
    
        print("Classification report for classifier {}:\n".format(clf))
        print("{}\n".format(classification_report(Y_test, Y_pred)))
        
        # disp = plot_confusion_matrix(clf, X_test, Y_test)
        # disp.figure_.suptitle("Confusion Matrix")
        # print(f"Confusion matrix:\n{disp.confusion_matrix}")
    
        plt.show()


#######################################

df = read_csv()

prep_df = prepare_df(df)
X, y = get_x_y(prep_df)

# define classification model # Using non-linear, since data appears non-linear

clf = svm.SVC()
# a_clf = svm.SVC()
# clf2 = MLPClassifier()
# a_clf2 = MLPClassifier()
# clf3 = KNeighborsClassifier()
# a_clf3 = KNeighborsClassifier()
# clf4 = svm.NuSVC()
# clf5 = DecisionTreeClassifier()

train_vowel(prep_df)