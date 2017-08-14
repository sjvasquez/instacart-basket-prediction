import os

from more_itertools import unique_everseen
import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('../data/processed/user_data.csv')

    user_ids = []
    department_ids = []
    eval_sets = []

    is_ordered_histories = []
    index_in_order_histories = []
    order_size_histories = []
    order_dow_histories = []
    order_hour_histories = []
    days_since_prior_order_histories = []
    order_number_histories = []
    num_products_from_department_histories = []

    longest = 0
    for _, row in df.iterrows():
        if _ % 10000 == 0:
            print _

        user_id = row['user_id']
        eval_set = row['eval_set']
        departments = row['department_ids']

        departments, next_departments = ' '.join(departments.split()[:-1]), departments.split()[-1]

        department_set = set([int(j) for i in departments.split() for j in i.split('_')])
        next_department_set = set([int(i) for i in next_departments.split('_')])

        orders = [map(int, i.split('_')) for i in departments.split()]

        for department_id in department_set:

            user_ids.append(user_id)
            department_ids.append(department_id)
            eval_sets.append(eval_set)

            is_ordered = []
            index_in_order = []
            num_products_from_department = []
            order_size = []

            for order in orders:
                order_set = set(order)
                is_ordered.append(str(int(department_id in order_set)))
                unique_order = list(unique_everseen(order))
                index_in_order.append(str(unique_order.index(department_id) + 1) if department_id in order_set else '0')
                num_products_from_department.append(str(sum([i == department_id for i in order])))
                order_size.append(str(len(set(order))))

            is_ordered = ' '.join(is_ordered)
            num_products_from_department = ' '.join(num_products_from_department)
            order_size = ' '.join(order_size)
            index_in_order = ' '.join(index_in_order)

            is_ordered_histories.append(is_ordered)
            index_in_order_histories.append(index_in_order)
            order_size_histories.append(order_size)
            num_products_from_department_histories.append(num_products_from_department)

            order_dow_histories.append(row['order_dows'])
            order_hour_histories.append(row['order_hours'])

            days_since_prior_order_histories.append(row['days_since_prior_orders'])
            order_number_histories.append(row['order_numbers'])

    data = [
        user_ids,
        department_ids,
        department_ids,
        eval_sets,

        is_ordered_histories,
        index_in_order_histories,
        order_size_histories,
        order_dow_histories,
        order_hour_histories,
        days_since_prior_order_histories,
        order_number_histories,
        num_products_from_department_histories,
    ]
    columns = [
        'user_id',
        'department_id',
        'department_id',
        'eval_set',

        'is_ordered_history',
        'index_in_order_history',
        'order_size_history',
        'order_dow_history',
        'order_hour_history',
        'days_since_prior_order_history',
        'order_number_history',
        'num_products_from_department_history',
    ]
    if not os.path.isdir('../data/processed'):
        os.makedirs('../data/processed')

    df = pd.DataFrame(dict(zip(columns, data)))
    df.to_csv('../data/processed/department_data.csv', index=False)
