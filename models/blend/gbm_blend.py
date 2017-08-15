import gc
import os
import pprint as pp

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


order_id = np.load('data/order_id.npy')
product_id = np.load('data/product_id.npy')
features = np.load('data/features.npy')
feature_names = np.load('data/feature_names.npy')
label = np.load('data/label.npy')

product_df = pd.DataFrame(data=features, columns=feature_names)
product_df['order_id'] = order_id
product_df['product_id'] = product_id
product_df['label'] = label

del order_id, product_id, features, feature_names, label
gc.collect()

drop_cols = [i for i in product_df.columns if i.startswith('sgns') or i.startswith('nnmf')]
drop_cols += ['order_id', 'product_id', 'label']

# training
train_df = product_df[product_df['label'] != -1]
test_df = product_df[product_df['label'] == -1]
train_df, val_df = train_test_split(train_df, train_size=.99)
del product_df
gc.collect()

Y_train, Y_val = train_df['label'].astype(int).astype(float), val_df['label'].astype(int).astype(float)
X_train, X_val = train_df.drop(drop_cols, axis=1), val_df.drop(drop_cols, axis=1)
del train_df
gc.collect()

test_orders = test_df['order_id']
test_products = test_df['product_id']
test_labels = test_df['label']
X_test = test_df.drop(drop_cols, axis=1)
del test_df
gc.collect()

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'learning_rate': .02,
    'num_leaves': 32,
    'max_depth': 12,
    'feature_fraction': 0.35,
    'bagging_fraction': 0.9,
    'bagging_freq': 2,
}
rounds = 15000
d_train = lgb.Dataset(X_train, label=Y_train, silent=True)
d_valid = lgb.Dataset(X_val, label=Y_val, silent=True)
del X_train, X_val, Y_train, Y_val

valid_sets = [d_train, d_valid]
valid_names = ['train', 'valid']
gbdt = lgb.train(params, d_train, rounds, valid_sets=valid_sets, valid_names=valid_names, verbose_eval=20)

features = gbdt.feature_name()
importance = list(gbdt.feature_importance())
importance = zip(features, importance)
importance = sorted(importance, key=lambda x: x[1])
total = sum(j for i, j in importance)
importance = [(i, float(j)/total) for i, j in importance]
pp.pprint(importance)

test_preds = gbdt.predict(X_test, num_iteration=gbdt.best_iteration)

dirname = 'predictions_gbm'
if not os.path.isdir(dirname):
    os.makedirs(dirname)

np.save(os.path.join(dirname, 'order_ids.npy'), test_orders)
np.save(os.path.join(dirname, 'product_ids.npy'), test_products)
np.save(os.path.join(dirname, 'predictions.npy'), test_preds)
np.save(os.path.join(dirname, 'labels.npy'), test_labels)
