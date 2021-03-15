import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def is_outlier(s):
    
    lower_limit = s.mean() - (s.std() * 3)
    upper_limit = s.mean() + (s.std() * 3)
    
    return ~s.between(lower_limit, upper_limit)


def read_csv():
    
    # read csv file
    df = pd.read_csv('data\icelandic_vowels_f1_f2.csv')
    
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