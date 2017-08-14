import os

import numpy as np
import pandas as pd


def pad_1d(array, max_len):
    array = array[:max_len]
    length = len(array)
    padded = array + [0]*(max_len - len(array))
    return padded, length


if __name__ == '__main__':
    user_data = pd.read_csv('../../data/processed/user_data.csv')
    num_rows = len(user_data)

    order_size_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    reorder_size_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_number_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_dow_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_hour_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    days_since_prior_order_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)

    history_length = np.zeros(shape=[num_rows], dtype=np.int8)
    label = np.zeros(shape=[num_rows], dtype=np.int8)
    eval_set = np.zeros(shape=[num_rows], dtype='S5')
    user_id = np.zeros(shape=[num_rows], dtype=np.int64)

    for idx, row in user_data.iterrows():

        if idx % 10000 == 0:
            print idx

        products = row['product_ids']
        products, next_products = ' '.join(products.split()[:-1]), products.split()[-1]

        reorders = row['reorders']
        reorders, next_reorders = ' '.join(reorders.split()[:-1]), reorders.split()[-1]

        orders = [map(int, i.split('_')) for i in products.split()]
        reorders = [map(int, i.split('_')) for i in reorders.split()]

        next_reorders = map(int, next_reorders.split('_'))
        next_orders = map(int, next_products.split('_'))

        order_sizes = [len(i) for i in orders]
        reorder_sizes = [sum(i) for i in reorders]

        order_size_history[idx, :], history_length[idx] = pad_1d(order_sizes, 100)
        reorder_size_history[idx, :], _ = pad_1d(reorder_sizes, 100)
        order_number_history[idx, :], _ = pad_1d(map(int, row['order_numbers'].split()), 100)
        order_dow_history[idx, :], _ = pad_1d(map(int, row['order_dows'].split()), 100)
        order_hour_history[idx, :], _ = pad_1d(map(int, row['order_hours'].split()), 100)
        days_since_prior_order_history[idx, :], _ = pad_1d(map(int, row['days_since_prior_orders'].split()), 100)

        label[idx] = sum(next_reorders)
        eval_set[idx] = row['eval_set']
        user_id[idx] = row['user_id']

    if not os.path.isdir('data/'):
        os.makedirs('data/')

    np.save('data/order_size_history.npy', order_size_history)
    np.save('data/reorder_size_history.npy', reorder_size_history)
    np.save('data/order_number_history.npy', order_number_history)
    np.save('data/order_dow_history.npy', order_dow_history)
    np.save('data/order_hour_history.npy', order_hour_history)
    np.save('data/days_since_prior_order_history.npy', days_since_prior_order_history)
    np.save('data/history_length.npy', history_length)
    np.save('data/label.npy', label)
    np.save('data/eval_set.npy', eval_set)
    np.save('data/user_id.npy', user_id)
