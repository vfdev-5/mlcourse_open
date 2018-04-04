
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel


max_train_size = None
n_splits = 10

penalty = 'l2'
solver = 'lbfgs'
max_iter = 5000
Cs = np.logspace(-3, 1, 20)
class_weight = None


vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=25000)


yearmonth_scaler = StandardScaler()
start_hour_encoder = OneHotEncoder(dtype=np.int)
weekday_encoder = OneHotEncoder(dtype=np.int)
month_encoder = OneHotEncoder(dtype=np.int)

# ########################################################################


os.environ['JOBLIB_TEMP_FOLDER'] = "/tmp"

PATH_TO_DATA = '../../../data'
train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'), index_col='session_id')
test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'), index_col='session_id')

# Change site1, ..., site10 columns type to integer and fill NA-values with zeros
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Order by time1
train_df = train_df.sort_values(by='time1')

# Limit in time:
train_df = train_df[train_df['time1'] > '2013-11']

# Load websites dictionary
with open(os.path.join(PATH_TO_DATA, "site_dic.pkl"), "rb") as input_file:
    site_dict = pickle.load(input_file)

# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
print(u'Websites total:', sites_dict.shape[0])

y = train_df['target']


print("- Create count vectors")
full_df = pd.concat([train_df.drop('target', axis=1), test_df])
idx_split = train_df.shape[0]
full_sites = full_df[sites]
sites_flatten = full_sites.values.flatten()
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                sites_flatten,
                                range(0, sites_flatten.shape[0] + 10, 10)))[:, 1:]
train_sites_sparse = full_sites_sparse[:idx_split, :]
test_sites_sparse = full_sites_sparse[idx_split:, :]

train_sites_joined = train_df[sites].apply(lambda row: " ".join([str(v) for v in row]), axis=1)
test_sites_joined = test_df[sites].apply(lambda row: " ".join([str(v) for v in row]), axis=1)

vectorizer.fit(train_sites_joined)

train_sites_cvec = vectorizer.transform(train_sites_joined)
test_sites_cvec = vectorizer.transform(test_sites_joined)

# Join features:
# train_sites_cvec = hstack([train_sites_cvec, train_sites_sparse], format='csr')
# test_sites_cvec = hstack([test_sites_cvec, test_sites_sparse], format='csr')

print(train_sites_cvec.shape)
print(test_sites_cvec.shape)

print("Select features")
log_reg = LogisticRegression(random_state=17,
                             max_iter=max_iter, penalty=penalty, solver=solver,
                             class_weight=class_weight)
model = SelectFromModel(log_reg, threshold="mean")
model.fit(train_sites_cvec, y)
train_sites_cvec = model.transform(train_sites_cvec)
test_sites_cvec = model.transform(test_sites_cvec)
print(train_sites_cvec.shape)
print(test_sites_cvec.shape)


def get_auc_lr_cv(X, y, max_train_size=None, n_splits=7, seed=12, **kwargs):
    tscv = TimeSeriesSplit(n_splits, max_train_size=max_train_size)
    log_reg_cv = LogisticRegressionCV(cv=tscv, random_state=seed, verbose=1,
                                      scoring='roc_auc',
                                      n_jobs=-1, **kwargs)
    log_reg_cv.fit(X, y)
    return log_reg_cv, np.mean(log_reg_cv.scores_[1], axis=0), log_reg_cv.C_


def compute_features(df, sites_sparse, is_train=True):
    df.loc[:, 'yearmonth'] = df['time1'].apply(lambda x: x.year * 100 + x.month)
    df.loc[:, 'month'] = df['time1'].apply(lambda x: x.month)
    df.loc[:, 'weekday'] = df['time1'].apply(lambda x: x.weekday())
    df.loc[:, 'start_hour'] = df['time1'].apply(lambda x: x.hour)
    df.loc[:, 'morning'] = df['time1'].apply(lambda x: int(7 <= x.hour <= 11))
    df.loc[:, 'afternoon'] = df['time1'].apply(lambda x: int(12 <= x.hour <= 17))
    df.loc[:, 'evening'] = df['time1'].apply(lambda x: int(18 <= x.hour <= 22))

    if is_train:
        yearmonth_scaler.fit(df.loc[:, 'yearmonth'].values.reshape(-1, 1))
        start_hour_encoder.fit(df.loc[:, 'start_hour'].values.reshape(-1, 1))
        weekday_encoder.fit(df.loc[:, 'weekday'].values.reshape(-1, 1))
        month_encoder.fit(df.loc[:, 'month'].values.reshape(-1, 1))

    df.loc[:, 'yearmonth'] = yearmonth_scaler.transform(df.loc[:, 'yearmonth'].values.reshape(-1, 1))
    start_hour_ohe = start_hour_encoder.transform(df.loc[:, 'start_hour'].values.reshape(-1, 1))
    weekday_ohe = weekday_encoder.transform(df.loc[:, 'weekday'].values.reshape(-1, 1))
    month_ohe = month_encoder.transform(df.loc[:, 'month'].values.reshape(-1, 1))
    features = ['yearmonth', 'morning', 'afternoon', 'evening']

    X = hstack([sites_sparse, df[features].values, start_hour_ohe, weekday_ohe, month_ohe], format='csr')
    return X


print("CV log reg model")
X_train = compute_features(train_df, train_sites_cvec, is_train=True)
log_reg_cv, score, best_C = get_auc_lr_cv(X_train, y, max_iter=max_iter, n_splits=n_splits,
                                          Cs=Cs, penalty=penalty, solver=solver,
                                          class_weight=class_weight)
print(score, best_C)

print("Train on whole dataset")
log_reg = LogisticRegression(C=best_C[0], random_state=17,
                             max_iter=max_iter, penalty=penalty, solver=solver,
                             class_weight=class_weight)
log_reg.fit(X_train, y)

y_pred = log_reg.predict(X_train)
m = confusion_matrix(y, y_pred)
print("- Train confusion matrix: ", m)


def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index=np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


X_test = X_train = compute_features(test_df, test_sites_cvec, is_train=False)
y_test = log_reg.predict_proba(X_test)[:, 1]

score = np.max(score)
write_to_submission_file(y_test,
                         "assignment6_alice_submission_model_L4+countvec_v4a3_roc_auc={:.4f}_FN={}_FP={}_TP={}.csv"
                         .format(score, m[1, 0], m[0, 1], m[1, 1]))
