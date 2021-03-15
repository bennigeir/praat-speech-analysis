# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sn

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
# from sklearn.metrics import plot_confusion_matrix
# from sklearn.metrics import classification_report, plot_confusion_matrix

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


def augment_data(df, augment_df, drop):
    
    for i in range(-10,10):
        augmentation = df.copy(deep=True)
        augmentation[['F1', 'F2']] += i
        augment_df = pd.concat([augment_df, augmentation])

    augmented_X_train = augment_df.drop(columns=[drop])
    augmented_y_train = augment_df[[drop]].values.ravel()
    
    return augmented_X_train, augmented_y_train   


def train_kf(X, y):
    
    scores, scores2, scores3, scores4 = [], [], [], []
    augmented_scores, augmented_scores2, augmented_scores3, augmented_scores4 = [], [], [], []
    ensemble_scores=[]
    ensemble_augmented_scores=[]
    
    scores3, lr_scores, rfc_scores, dtc_scores, mlp_scores, ensemble_scores = [],[],[],[],[],[]
    scores3_aug, lr_scores_aug, rfc_scores_aug, dtc_scores_aug, mlp_scores_aug, ensemble_scores_aug = [],[],[],[],[],[]
    
    
    
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
                                                            augment_training,
                                                            'native_isl')
    
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
        
        mnb_fitted = mnb.fit(X_train, Y_train)
        scores4.append(mnb_fitted.score(X_test, Y_test))
        mnb_fitted = mnb.fit(augmented_X_train, augmented_y_train)
        augmented_scores4.append(mnb_fitted.score(X_test, Y_test))
        
        
        lr_fitted = lr.fit(X_train, Y_train)
        lr_score = (lr_fitted.score(X_test, Y_test))
        lr_scores.append(lr_fitted.score(X_test, Y_test))
        lr_fitted = lr.fit(augmented_X_train, augmented_y_train)
        lr_augmented_score = (lr_fitted.score(X_test, Y_test))
        lr_scores_aug.append(lr_fitted.score(X_test, Y_test))
        
        rfc_fitted = rfc.fit(X_train, Y_train)
        rfc_score = (rfc_fitted.score(X_test, Y_test))
        rfc_scores.append(rfc_fitted.score(X_test, Y_test))
        rfc_fitted = rfc.fit(augmented_X_train, augmented_y_train)
        rfc_augmented_score = (rfc_fitted.score(X_test, Y_test))
        rfc_scores_aug.append(rfc_fitted.score(X_test, Y_test))
        
        dtc_fitted = dtc.fit(X_train, Y_train)
        dtc_score = (dtc_fitted.score(X_test, Y_test))
        dtc_scores.append(dtc_fitted.score(X_test, Y_test))
        dtc_fitted = dtc.fit(augmented_X_train, augmented_y_train)
        dtc_augmented_score = (dtc_fitted.score(X_test, Y_test))
        dtc_scores_aug.append(dtc_fitted.score(X_test, Y_test))
        
        mlp_fitted = mlp.fit(X_train, Y_train)
        mlp_score = (mlp_fitted.score(X_test, Y_test))
        mlp_scores.append(mlp_fitted.score(X_test, Y_test))
        mlp_fitted = mlp.fit(augmented_X_train, augmented_y_train)
        mlp_augmented_score = (mlp_fitted.score(X_test, Y_test))
        mlp_scores_aug.append(mlp_fitted.score(X_test, Y_test))
                
        
        # ensemble addition
        ensemble_fitted = ensemble.fit(X_train, Y_train)
        ensemble_scores.append(ensemble_fitted.score(X_test, Y_test)) 
        ensemble_fitted = ensemble.fit(augmented_X_train, augmented_y_train)
        ensemble_augmented_scores.append(ensemble_fitted.score(X_test, Y_test))
        
        preds = ensemble.predict(X_test)
        
        
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
        print(20*'-')
        
        
        print('KNeighboursClassifier scores: ', scores3)
        print('KNeighboursClassifier scores augmented: ', augmented_scores3)
        
        print('LogisticRegression scores: ', lr_scores)
        print('LogisticRegression scores augmented: ', lr_scores_aug)
        
        print('RandomForestClassifier scores: ', rfc_scores)
        print('RandomForestClassifier scores augmented: ', rfc_scores_aug)
        
        print('DecisionTreeClassifier scores: ', dtc_scores)
        print('DecisionTreeClassifier scores augmented: ', dtc_scores_aug)
        
        print('MLPClassifier scores: ', mlp_scores)
        print('MLPClassifier scores augmented: ', mlp_scores_aug)
        
        
        # ensemble addition
        print('Ensemble scores:', ensemble_scores)
        print('Ensemble scores augmented:', ensemble_scores_aug)
        

                
    print('SVC scores:', scores)
    print('SVC scores augmented:', augmented_scores)
    print('MLPClassifier scores:', scores2)
    print('MLPClassifier scores augmented:', augmented_scores2)
    print('KNeighboursClassifier scores: ', scores4)
    print('KNeighboursClassifier scores augmented: ', augmented_scores3)
    
    print('MNBClassifier scores: ', scores3)
    print('MNBClassifier scores augmented: ', augmented_scores4)
    
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
                                                            augment_training,
                                                            'native_isl')
    
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
        
        lr_fitted = lr.fit(X_train, Y_train)
        lr_score = (lr_fitted.score(X_test, Y_test))
        lr_fitted = lr.fit(augmented_X_train, augmented_y_train)
        lr_augmented_score = (lr_fitted.score(X_test, Y_test))
        
        rfc_fitted = rfc.fit(X_train, Y_train)
        rfc_score = (rfc_fitted.score(X_test, Y_test))
        rfc_fitted = rfc.fit(augmented_X_train, augmented_y_train)
        rfc_augmented_score = (rfc_fitted.score(X_test, Y_test))
        
        dtc_fitted = dtc.fit(X_train, Y_train)
        dtc_score = (dtc_fitted.score(X_test, Y_test))
        dtc_fitted = dtc.fit(augmented_X_train, augmented_y_train)
        dtc_augmented_score = (dtc_fitted.score(X_test, Y_test))
        
        mlp_fitted = mlp.fit(X_train, Y_train)
        mlp_score = (mlp_fitted.score(X_test, Y_test))
        mlp_fitted = mlp.fit(augmented_X_train, augmented_y_train)
        mlp_augmented_score = (mlp_fitted.score(X_test, Y_test))
        
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
        print('LogisticRegression scores augmented: ', lr_augmented_score)
        
        print('RandomForestClassifier scores: ', rfc_score)
        print('RandomForestClassifier scores augmented: ', rfc_augmented_score)
        
        print('DecisionTreeClassifier scores: ', dtc_score)
        print('DecisionTreeClassifier scores augmented: ', dtc_augmented_score)
        
        print('MLPClassifier scores: ', mlp_score)
        print('MLPClassifier scores augmented: ', mlp_augmented_score)
        
        
        # ensemble addition
        print('Ensemble scores:', ensemble_scores)
        print('Ensemble scores augmented:', ensemble_augmented_scores)


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

# define classification model
clf = svm.SVC()
clf2 = MLPClassifier(solver='lbfgs',
                     alpha=1e-5,
                     hidden_layer_sizes=(5, 2),
                     random_state=1, max_iter=200)
clf3 = KNeighborsClassifier()
mnb = MultinomialNB()

#######################################

lr = LogisticRegression()
rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
knc = KNeighborsClassifier()
mlp = MLPClassifier()

# ensemble
estimator = []
estimator.append(('MLP', MLPClassifier(solver='lbfgs',
                                       alpha=1e-5,
                                       hidden_layer_sizes=(5, 2),
                                       random_state=1,
                                       max_iter=200))) 
estimator.append(('DTC', DecisionTreeClassifier())) 
estimator.append(('RFC', RandomForestClassifier())) 
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

# train_name(prep_df)

#######################################

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