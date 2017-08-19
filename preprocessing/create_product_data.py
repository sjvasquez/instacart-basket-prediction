import os

import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('../data/processed/user_data.csv')

    products = pd.read_csv('../data/raw/products.csv')
    product_to_aisle = dict(zip(products['product_id'], products['aisle_id']))
    product_to_department = dict(zip(products['product_id'], products['department_id']))
    product_to_name = dict(zip(products['product_id'], products['product_name']))

    user_ids = []
    product_ids = []
    aisle_ids = []
    department_ids = []
    product_names = []
    eval_sets = []

    is_ordered_histories = []
    index_in_order_histories = []
    order_size_histories = []
    reorder_size_histories = []
    order_dow_histories = []
    order_hour_histories = []
    days_since_prior_order_histories = []
    order_number_histories = []

    labels = []

    longest = 0
    for _, row in df.iterrows():
        if _ % 10000 == 0:
            print _

        user_id = row['user_id']
        eval_set = row['eval_set']
        products = row['product_ids']

        products, next_products = ' '.join(products.split()[:-1]), products.split()[-1]

        reorders = row['reorders']
        reorders, next_reorders = ' '.join(reorders.split()[:-1]), reorders.split()[-1]

        product_set = set([int(j) for i in products.split() for j in i.split('_')])
        next_product_set = set([int(i) for i in next_products.split('_')])

        orders = [map(int, i.split('_')) for i in products.split()]
        reorders = [map(int, i.split('_')) for i in reorders.split()]
        next_reorders = map(int, next_reorders.split('_'))

        for product_id in product_set:

            user_ids.append(user_id)
            product_ids.append(product_id)
            labels.append(int(product_id in next_product_set) if eval_set == 'train' else -1)

            aisle_ids.append(product_to_aisle[product_id])
            department_ids.append(product_to_department[product_id])
            product_names.append(product_to_name[product_id])
            eval_sets.append(eval_set)

            is_ordered = []
            index_in_order = []
            order_size = []
            reorder_size = []

            prior_products = set()
            for order in orders:
                is_ordered.append(str(int(product_id in order)))
                index_in_order.append(str(order.index(product_id) + 1) if product_id in order else '0')
                order_size.append(str(len(order)))
                reorder_size.append(str(len(prior_products & set(order))))
                prior_products |= set(order)

            is_ordered = ' '.join(is_ordered)
            index_in_order = ' '.join(index_in_order)
            order_size = ' '.join(order_size)
            reorder_size = ' '.join(reorder_size)

            is_ordered_histories.append(is_ordered)
            index_in_order_histories.append(index_in_order)
            order_size_histories.append(order_size)
            reorder_size_histories.append(reorder_size)
            order_dow_histories.append(row['order_dows'])
            order_hour_histories.append(row['order_hours'])
            days_since_prior_order_histories.append(row['days_since_prior_orders'])
            order_number_histories.append(row['order_numbers'])

        user_ids.append(user_id)
        product_ids.append(0)
        labels.append(int(max(next_reorders) == 0) if eval_set == 'train' else -1)

        aisle_ids.append(0)
        department_ids.append(0)
        product_names.append(0)
        eval_sets.append(eval_set)

        is_ordered = []
        index_in_order = []
        order_size = []
        reorder_size = []

        for reorder in reorders:
            is_ordered.append(str(int(max(reorder) == 0)))
            index_in_order.append(str(0))
            order_size.append(str(len(reorder)))
            reorder_size.append(str(sum(reorder)))

        is_ordered = ' '.join(is_ordered)
        index_in_order = ' '.join(index_in_order)
        order_size = ' '.join(order_size)
        reorder_size = ' '.join(reorder_size)

        is_ordered_histories.append(is_ordered)
        index_in_order_histories.append(index_in_order)
        order_size_histories.append(order_size)
        reorder_size_histories.append(reorder_size)
        order_dow_histories.append(row['order_dows'])
        order_hour_histories.append(row['order_hours'])
        days_since_prior_order_histories.append(row['days_since_prior_orders'])
        order_number_histories.append(row['order_numbers'])

    data = [
        user_ids,
        product_ids,
        aisle_ids,
        department_ids,
        product_names,
        is_ordered_histories,
        index_in_order_histories,
        order_size_histories,
        reorder_size_histories,
        order_dow_histories,
        order_hour_histories,
        days_since_prior_order_histories,
        order_number_histories,
        labels,
        eval_sets
    ]
    columns = [
        'user_id',
        'product_id',
        'aisle_id',
        'department_id',
        'product_name',
        'is_ordered_history',
        'index_in_order_history',
        'order_size_history',
        'reorder_size_history',
        'order_dow_history',
        'order_hour_history',
        'days_since_prior_order_history',
        'order_number_history',
        'label',
        'eval_set'
    ]
    if not os.path.isdir('../data/processed'):
        os.makedirs('../data/processed')

    df = pd.DataFrame(dict(zip(columns, data)))
    df.to_csv('../data/processed/product_data.csv', index=False)
