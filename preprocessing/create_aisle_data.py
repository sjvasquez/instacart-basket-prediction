import os

from  more_itertools import unique_everseen
import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('../data/processed/user_data.csv')

    products = pd.read_csv('../data/raw/products.csv')
    aisle_to_department = dict(zip(products['aisle_id'], products['department_id']))

    user_ids = []
    aisle_ids = []
    department_ids = []
    eval_sets = []

    is_ordered_histories = []
    index_in_order_histories = []
    order_size_histories = []
    order_dow_histories = []
    order_hour_histories = []
    days_since_prior_order_histories = []
    order_number_histories = []
    num_products_from_aisle_histories = []

    longest = 0
    for _, row in df.iterrows():
        if _ % 10000 == 0:
            print _

        user_id = row['user_id']
        eval_set = row['eval_set']
        aisles = row['aisle_ids']

        aisles, next_aisles = ' '.join(aisles.split()[:-1]), aisles.split()[-1]

        aisle_set = set([int(j) for i in aisles.split() for j in i.split('_')])
        next_aisle_set = set([int(i) for i in next_aisles.split('_')])

        orders = [map(int, i.split('_')) for i in aisles.split()]

        for aisle_id in aisle_set:

            user_ids.append(user_id)
            aisle_ids.append(aisle_id)

            department_ids.append(aisle_to_department[aisle_id])
            eval_sets.append(eval_set)

            is_ordered = []
            index_in_order = []
            num_products_from_aisle = []
            order_size = []

            for order in orders:
                order_set = set(order)
                is_ordered.append(str(int(aisle_id in order_set)))
                unique_order = list(unique_everseen(order))
                index_in_order.append(str(unique_order.index(aisle_id) + 1) if aisle_id in order_set else '0')
                num_products_from_aisle.append(str(sum([i == aisle_id for i in order])))
                order_size.append(str(len(set(order))))

            is_ordered = ' '.join(is_ordered)
            num_products_from_aisle = ' '.join(num_products_from_aisle)
            order_size = ' '.join(order_size)
            index_in_order = ' '.join(index_in_order)

            is_ordered_histories.append(is_ordered)
            index_in_order_histories.append(index_in_order)
            order_size_histories.append(order_size)
            num_products_from_aisle_histories.append(num_products_from_aisle)

            order_dow_histories.append(row['order_dows'])
            order_hour_histories.append(row['order_hours'])

            days_since_prior_order_histories.append(row['days_since_prior_orders'])
            order_number_histories.append(row['order_numbers'])

    data = [
        user_ids,
        aisle_ids,
        department_ids,
        eval_sets,

        is_ordered_histories,
        index_in_order_histories,
        order_size_histories,
        order_dow_histories,
        order_hour_histories,
        days_since_prior_order_histories,
        order_number_histories,
        num_products_from_aisle_histories,
    ]
    columns = [
        'user_id',
        'aisle_id',
        'department_id',
        'eval_set',

        'is_ordered_history',
        'index_in_order_history',
        'order_size_history',
        'order_dow_history',
        'order_hour_history',
        'days_since_prior_order_history',
        'order_number_history',
        'num_products_from_aisle_history',
    ]
    if not os.path.isdir('../data/processed'):
        os.makedirs('../data/processed')

    df = pd.DataFrame(dict(zip(columns, data)))
    df.to_csv('../data/processed/aisle_data.csv', index=False)
