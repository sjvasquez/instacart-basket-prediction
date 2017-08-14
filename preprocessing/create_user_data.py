import os

import pandas as pd


def parse_order(x):
    series = pd.Series()

    series['products'] = '_'.join(x['product_id'].values.astype(str).tolist())
    series['reorders'] = '_'.join(x['reordered'].values.astype(str).tolist())
    series['aisles'] = '_'.join(x['aisle_id'].values.astype(str).tolist())
    series['departments'] = '_'.join(x['department_id'].values.astype(str).tolist())

    series['order_number'] = x['order_number'].iloc[0]
    series['order_dow'] = x['order_dow'].iloc[0]
    series['order_hour'] = x['order_hour_of_day'].iloc[0]
    series['days_since_prior_order'] = x['days_since_prior_order'].iloc[0]

    return series


def parse_user(x):
    parsed_orders = x.groupby('order_id', sort=False).apply(parse_order)

    series = pd.Series()

    series['order_ids'] = ' '.join(parsed_orders.index.map(str).tolist())
    series['order_numbers'] = ' '.join(parsed_orders['order_number'].map(str).tolist())
    series['order_dows'] = ' '.join(parsed_orders['order_dow'].map(str).tolist())
    series['order_hours'] = ' '.join(parsed_orders['order_hour'].map(str).tolist())
    series['days_since_prior_orders'] = ' '.join(parsed_orders['days_since_prior_order'].map(str).tolist())

    series['product_ids'] = ' '.join(parsed_orders['products'].values.astype(str).tolist())
    series['aisle_ids'] = ' '.join(parsed_orders['aisles'].values.astype(str).tolist())
    series['department_ids'] = ' '.join(parsed_orders['departments'].values.astype(str).tolist())
    series['reorders'] = ' '.join(parsed_orders['reorders'].values.astype(str).tolist())

    series['eval_set'] = x['eval_set'].values[-1]

    return series

if __name__ == '__main__':
    orders = pd.read_csv('../data/raw/orders.csv')
    prior_products = pd.read_csv('../data/raw/order_products__prior.csv')
    train_products = pd.read_csv('../data/raw/order_products__train.csv')
    order_products = pd.concat([prior_products, train_products], axis=0)
    products = pd.read_csv('../data/raw/products.csv')

    df = orders.merge(order_products, how='left', on='order_id')
    df = df.merge(products, how='left', on='product_id')
    df['days_since_prior_order'] = df['days_since_prior_order'].fillna(0).astype(int)
    null_cols = ['product_id', 'aisle_id', 'department_id', 'add_to_cart_order', 'reordered']
    df[null_cols] = df[null_cols].fillna(0).astype(int)

    if not os.path.isdir('../data/processed'):
        os.makedirs('../data/processed')

    user_data = df.groupby('user_id', sort=False).apply(parse_user).reset_index()
    user_data.to_csv('../data/processed/user_data.csv', index=False)
