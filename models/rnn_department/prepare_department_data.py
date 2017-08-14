import os

import pandas as pd
import numpy as np


def pad_1d(array, max_len):
    array = array[:max_len]
    length = len(array)
    padded = array + [0]*(max_len - len(array))
    return padded, length


if __name__ == '__main__':
    department_data = pd.read_csv('../../data/processed/department_data.csv')
    num_rows = len(department_data)

    user_id = np.zeros(shape=[num_rows], dtype=np.int32)
    department_id = np.zeros(shape=[num_rows], dtype=np.int16)
    eval_set = np.zeros(shape=[num_rows], dtype='S5')

    is_ordered_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    index_in_order_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_size_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_dow_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_hour_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    days_since_prior_order_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_number_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    num_products_from_department_history = np.zeros(shape=[num_rows, 100], dtype=np.int16)

    history_length = np.zeros(shape=[num_rows], dtype=np.int8)

    for i, row in department_data.iterrows():
        if i % 10000 == 0:
            print i, num_rows

        user_id[i] = row['user_id']
        department_id[i] = row['department_id']
        eval_set[i] = row['eval_set']

        is_ordered_history[i, :], history_length[i] = pad_1d(map(int, row['is_ordered_history'].split()), 100)
        index_in_order_history[i, :], _ = pad_1d(map(int, row['index_in_order_history'].split()), 100)
        order_size_history[i, :], _ = pad_1d(map(int, row['order_size_history'].split()), 100)
        order_dow_history[i, :], _ = pad_1d(map(int, row['order_dow_history'].split()), 100)
        order_hour_history[i, :], _ = pad_1d(map(int, row['order_hour_history'].split()), 100)
        days_since_prior_order_history[i, :], _ = pad_1d(map(int, row['days_since_prior_order_history'].split()), 100)
        order_number_history[i, :], _ = pad_1d(map(int, row['order_number_history'].split()), 100)
        num_products_from_department_history[i, :], _ = pad_1d(map(int, row['num_products_from_department_history'].split()), 100)

    if not os.path.isdir('data'):
        os.makedirs('data')

    np.save('data/user_id.npy', user_id)
    np.save('data/department_id.npy', department_id)
    np.save('data/eval_set.npy', eval_set)

    np.save('data/is_ordered_history.npy', is_ordered_history)
    np.save('data/index_in_order_history.npy', index_in_order_history)
    np.save('data/order_dow_history.npy', order_dow_history)
    np.save('data/order_hour_history.npy', order_hour_history)
    np.save('data/days_since_prior_order_history.npy', days_since_prior_order_history)
    np.save('data/order_size_history.npy', order_size_history)
    np.save('data/order_number_history.npy', order_number_history)
    np.save('data/num_products_from_department_history.npy', num_products_from_department_history)

    np.save('data/history_length.npy', history_length)
