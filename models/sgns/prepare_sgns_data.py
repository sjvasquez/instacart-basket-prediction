import os

import numpy as np
import pandas as pd


if __name__ == '__main__':
    user_data = pd.read_csv('../../data/processed/user_data.csv')

    x = []
    y = []
    for _, row in user_data.iterrows():
        if _ % 10000 == 0:
            print _

        user_id = row['user_id']
        products = row['product_ids']
        products = ' '.join(products.split()[:-1])
        for order in products.split():
            items = order.split('_')
            for i in range(len(items)):
                for j in range(max(0, i - 2), min(i + 3, len(items))):
                    if i != j:
                        x.append(int(items[j]))
                        y.append(int(items[i]))


    x = np.array(x)
    y = np.array(y)

    if not os.path.isdir('data'):
        os.makedirs('data')

    np.save('data/x.npy', x)
    np.save('data/y.npy', y)
