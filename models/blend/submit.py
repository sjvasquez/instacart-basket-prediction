from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from f1_optimizer import F1Optimizer


def select_products(x):
    series = pd.Series()

    true_products = [str(prod) if prod != 0 else 'None' for prod in x['product_id'][x['label'] > 0.5].values]
    true_products = ' '.join(true_products) if true_products else 'None'

    prod_preds_dict = dict(zip(x['product_id'].values, x['prediction'].values))
    none_prob = prod_preds_dict.get(0, None)
    del prod_preds_dict[0]

    other_products = np.array(prod_preds_dict.keys())
    other_probs = np.array(prod_preds_dict.values())

    idx = np.argsort(-1*other_probs)
    other_products = other_products[idx]
    other_probs = other_probs[idx]

    opt = F1Optimizer.maximize_expectation(other_probs, none_prob)
    best_prediction = ['None'] if opt[1] else []
    best_prediction += list(other_products[:opt[0]])

    predicted_products = ' '.join(map(str, best_prediction)) if best_prediction else 'None'

    series['products'] = predicted_products
    series['true_products'] = true_products

    return true_products, predicted_products


gbm_df = pd.DataFrame({
    'order_id': np.load('predictions_gbm/order_ids.npy'),
    'product_id': np.load('predictions_gbm/product_ids.npy'),
    'prediction_gbm': np.load('predictions_gbm/predictions.npy'),
    'label': np.load('predictions_gbm/labels.py')
})

nn_df = pd.DataFrame({
    'order_id': np.load('predictions_nn/order_ids.npy'),
    'product_id': np.load('predictions_nn/product_ids.npy'),
    'prediction_nn': np.load('predictions_nn/predictions.npy'),
})
pred_df = gbm_df.merge(nn_df, how='left', on=['order_id', 'product_id'])
pred_df['prediction'] = .9*pred_df['prediction_gbm'] + .1*pred_df['prediction_nn']

gb = pred_df.groupby('order_id')
dfs, order_ids = zip(*[(df, key) for key, df in gb])
p = Pool(cpu_count())
true_products, predicted_products = zip(*p.map(select_products, dfs))

pred_df = pd.DataFrame({'products': predicted_products, 'true_products': true_products, 'order_id': order_ids})
pred_df[['order_id', 'products']].to_csv('sub.csv', index=False)
