
import warnings
import os
import numpy as np
import pandas as pd
import pickle

from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel


def tokenizer(s):
    words = s.split(" ")
    return words


max_train_size = None
n_splits = 5
vectorizer = TfidfVectorizer(max_features=5000000, ngram_range=(1, 4), tokenizer=tokenizer)
tscv = TimeSeriesSplit(n_splits, max_train_size=max_train_size)
Cs = 10
penalty = 'l2'
solver = 'liblinear'
max_iter = 5000

class_weight = {
    0: 1,
    1: 3,
}

max_train_size = None


def to_daytime(hour):
    if 7 <= hour < 9:
        return 0
    elif 9 <= hour < 11:
        return 1
    elif 11 <= hour < 14:
        return 2
    elif 14 <= hour < 17:
        return 3
    elif 17 <= hour < 20:
        return 4
    elif 20 <= hour < 24:
        return 5
    elif 0 <= hour < 4:
        return 6
    elif 4 <= hour < 7:
        return 7
    else:
        return -1


def is_not_weekend(weekday):
    return 0 if 5 <= weekday <= 6 else 1


def is_saturday(weekday):
    return 1 if 5 == weekday else 0


def is_sunday(weekday):
    return 1 if 6 == weekday else 0


# ########################################################################


os.environ['JOBLIB_TEMP_FOLDER'] = "/tmp"

PATH_TO_DATA = '../../../data'
train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'), index_col='session_id')
test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'), index_col='session_id')

# Order by time1
train_df = train_df.sort_values(by='time1')

# Change site1, ..., site10 columns type to integer and fill NA-values with zeros
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')


# Load websites dictionary
with open(os.path.join(PATH_TO_DATA, "site_dic.pkl"), "rb") as input_file:
    site_dict = pickle.load(input_file)

# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
print(u'Websites total:', sites_dict.shape[0])

y = train_df['target']

print("- Create full sites")
full_df = pd.concat([train_df.drop('target', axis=1), test_df], ignore_index=True)
sites_full = full_df[sites]
train_indices = sites_full.index[:len(train_df)]
test_indices = sites_full.index[len(train_df):]

for site_id in sites:
    print(" . ", end='')
    sites_full.loc[:, site_id] = sites_full[site_id].apply(lambda x: sites_dict['site'][x] + " " if x > 0 else "")
print('')

sites_full_aligned = sites_full.loc[:, :].sum(axis=1)
sites_full_aligned = sites_full_aligned.apply(lambda x: x[:-1])
sites_full_aligned = sites_full_aligned.str.replace('.com', '').str.replace('www.', '')

print("- Fit vectorizer")
vectorizer.fit(sites_full_aligned)

train_sites_tfidf = vectorizer.transform(sites_full_aligned[train_indices])
test_sites_tfidf = vectorizer.transform(sites_full_aligned[test_indices])


def compute_timebased_features(df):
    df = df.copy()
    times = ['time%s' % i for i in range(1, 11)]
    for time_id in times:
        df[time_id] = pd.to_datetime(df[time_id])

    df.loc[:, 'daytime1'] = df.loc[:, 'time1'].apply(lambda x: to_daytime(x.hour))
    res1 = pd.get_dummies(df['daytime1'], prefix='daytime')
    res1['daytime_0'] = ~res1['daytime_0']
    res1['daytime_1'] = ~res1['daytime_1']
    res1['daytime_2'] = ~res1['daytime_2']
    res1['daytime_5'] = ~res1['daytime_5']

    df.loc[:, 'h'] = df.loc[:, 'time1'].apply(lambda x: x.hour)
    res2 = pd.get_dummies(df['h'], prefix='h')
    res2 = ~res2
    res2['h_12'] = ~res2['h_12']
    res2['h_16'] = ~res2['h_16']
    res2['h_17'] = ~res2['h_17']
    res2['h_18'] = ~res2['h_18']

    df.loc[:, 'hm'] = df.loc[:, 'time1'].apply(lambda x: x.hour * 100 + x.minute)
    # PROBLEM HERE WHEN APPLIED ON TEST DATA
    df.loc[:, 'hm'] = StandardScaler().fit_transform(df['hm'].values.reshape(-1, 1))

    df.loc[:, 'wd'] = df.loc[:, 'time1'].apply(lambda x: x.weekday())
    res3 = pd.get_dummies(df['wd'], prefix='wd')
    res3['wd_2'] = ~res3['wd_2']
    res3['wd_4'] = ~res3['wd_4']
    res3['wd_5'] = ~res3['wd_5']
    res3['wd_6'] = ~res3['wd_6']

    df.loc[:, 'start_hour'] = df['time1'].apply(lambda x: x.hour)
    df.loc[:, 'start_month'] = df['time1'].apply(lambda x: x.month)

    df.loc[:, 'is_not_weekend'] = df.loc[:, 'time1'].apply(lambda x: is_not_weekend(x.weekday()))
    df.loc[:, 'is_not_saturday'] = df.loc[:, 'time1'].apply(lambda x: ~is_saturday(x.weekday()))
    df.loc[:, 'is_not_sunday'] = df.loc[:, 'time1'].apply(lambda x: ~is_sunday(x.weekday()))

    df.loc[:, 'session_duration'] = (df.loc[:, 'time10'] - df.loc[:, 'time1']).apply(lambda x: -x.seconds)
    df['session_duration'].fillna(0, inplace=True)
    # PROBLEM HERE WHEN APPLIED ON TEST DATA
    df.loc[:, 'session_duration'] = StandardScaler().fit_transform(df['session_duration'].values.reshape(-1, 1))
    features = ['session_duration', 'is_not_saturday',
                'is_not_sunday', 'hm',
                'start_hour', 'start_month']
    return pd.concat([df[features], res1, res2, res3], axis=1)


def add_timebased_features(X, df):
    df = compute_timebased_features(df)
    X = hstack([X, df])
    return X


X = add_timebased_features(train_sites_tfidf, train_df)
print(X.shape)

print("- Feature selection")
log_reg = LogisticRegression(penalty=penalty, solver=solver,
                             class_weight=class_weight, max_iter=max_iter,
                             random_state=42, verbose=True, n_jobs=-1)
log_reg.fit(X, y)
model = SelectFromModel(log_reg, prefit=True)
X_new = model.transform(X)
print(X_new.shape)
X = X_new

print("- CV model")
log_reg_cv = LogisticRegressionCV(Cs=Cs, penalty=penalty,
                                  cv=tscv, scoring='roc_auc', solver=solver,
                                  class_weight=class_weight, max_iter=max_iter,
                                  random_state=42, verbose=False, n_jobs=-1)
log_reg_cv.fit(X, y)

print(np.mean(log_reg_cv.scores_[1], axis=0), log_reg_cv.C_)
score = np.max(np.mean(log_reg_cv.scores_[1], axis=0))
print("max roc_auc:", score)

print("- Train on all data")
log_reg = LogisticRegression(C=log_reg_cv.C_[0], penalty=penalty, solver=solver,
                             class_weight=class_weight, max_iter=max_iter,
                             random_state=42, verbose=False, n_jobs=-1)
log_reg.fit(X, y)
y_pred = log_reg.predict(X)
m = confusion_matrix(y, y_pred)
print("- Train confusion matrix: ", m)

print("- Predictions")
X_test = add_timebased_features(test_sites_tfidf, test_df)
X_test = model.transform(X_test)
test_probas = log_reg.predict_proba(X_test)
test_probas = test_probas[:, 1]


def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index=np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


write_to_submission_file(test_probas, "assignment6_alice_submission_model8_roc_auc={:.4f}_FN={}_FP={}_TP={}.csv"
                         .format(score, m[1, 0], m[0, 1], m[1, 1]))
