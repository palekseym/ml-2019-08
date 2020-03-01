import multiprocessing as mp
import tqdm
import time
import csv
import time
import pandas as pd
import os
import pickle
import numpy as np
from collections import Counter, defaultdict

import glob

def gen_dict_for_matrix(purchase):
    # наличие продуктов в транзакциях
    counter_product = purchase.product_id.value_counts()

    # потрачено денег на продукт
    counter_product_sum = purchase.groupby('product_id').trn_sum_from_iss.sum()

    # количество транзакций по магазинам
    counter_store = purchase.groupby(['store_id']).transaction_id.unique().agg(len)

    # сумма покупок по магазину
    counter_store_sum = purchase.groupby(['store_id', 'transaction_id']).purchase_sum.max().unstack().sum(axis=1)

    # Частота покупки бренда по всем транзакциям клиента
    counter_brand = Counter(purchase.groupby(['transaction_id']).brand_id.value_counts().index.get_level_values(1))

    # суммы потраченые на бренд
    counter_brand_sum = purchase.groupby(['brand_id']).trn_sum_from_iss.sum()

    return (
        counter_product.to_dict(),
        counter_product_sum.to_dict(),
        counter_store.to_dict(),
        counter_store_sum.to_dict(),
        counter_brand,
        counter_brand_sum.to_dict()
    )


def loop_q(items):
    while True:
        for i in items:
            yield i


def worker(q_users, counter):
    users = defaultdict(bool)
    purchases = []
    while True:
        user_id = q_users.get()

        if user_id is None:
            break

        users[user_id] = True

    pandas_reader = pd.read_csv('data/purchases.csv', chunksize=1000000, iterator=True)

    data = next(pandas_reader)
    for chunk in pandas_reader:
        data = pd.concat([data, chunk], ignore_index=True)

    index_for_del = data.client_id[(lambda clients: [not (users[i]) for i in clients])].index
    data.drop(index_for_del, inplace=True)

    products = pd.read_csv('data/products.csv', index_col='product_id')
    products.brand_id.fillna('NaN', inplace=True)

    data = pd.merge(data, products.reset_index(level=0)[['brand_id', 'product_id']], how='left',
                         left_on='product_id', right_on='product_id')

    with open(f'data/matrix/matrix_product_{os.getpid()}.pkl', 'wb') as matrix_product, \
            open(f'data/matrix/matrix_product_sum_{os.getpid()}.pkl', 'wb') as matrix_product_sum, \
            open(f'data/matrix/matrix_store_{os.getpid()}.pkl', 'wb') as matrix_store, \
            open(f'data/matrix/matrix_store_sum_{os.getpid()}.pkl', 'wb') as matrix_store_sum, \
            open(f'data/matrix/matrix_brand_{os.getpid()}.pkl', 'wb') as matrix_brand, \
            open(f'data/matrix/matrix_brand_sum_{os.getpid()}.pkl', 'wb') as matrix_brand_sum:

        pkl_product = []
        pkl_product_sum = []
        pkl_store = []
        pkl_store_sum = []
        pkl_brand = []
        pkl_brand_sum = []

        for group in data.groupby('client_id'):
            client_id, df_client = group
            row_product, row_product_sum, row_store, row_store_sum, row_brand, row_brand_sum \
                = gen_dict_for_matrix(df_client)

            row_product['client_id'] = client_id
            row_product_sum['client_id'] = client_id
            row_store['client_id'] = client_id
            row_store_sum['client_id'] = client_id
            row_brand['client_id'] = client_id
            row_brand_sum['client_id'] = client_id

            pkl_product.append(row_product)
            pkl_product_sum.append(row_product_sum)
            pkl_store.append(row_store)
            pkl_store_sum.append(row_store_sum)
            pkl_brand.append(row_brand)
            pkl_brand_sum.append(row_brand_sum)

            counter.value += 1

        pickle.dump(pkl_product, matrix_product)
        pickle.dump(pkl_product_sum, matrix_product_sum)
        pickle.dump(pkl_store, matrix_store)
        pickle.dump(pkl_store_sum, matrix_store_sum)
        pickle.dump(pkl_brand, matrix_brand)
        pickle.dump(pkl_brand_sum, matrix_brand_sum)



    print("END")


if __name__ == '__main__':
    CPU = 4

    workers = []
    q = []
    v = []
    
    for i in range(CPU):
        q.append(mp.Queue())
        v.append(mp.Value('i', 0))
    
    loop = loop_q(q)
    
    with open('data/clients.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            next(loop).put(row['client_id'])

    qsize_all = sum([i.qsize() for i in q])

    [i.put(None) for i in q]

    for work_i in range(CPU):
        work = mp.Process(target=worker, args=(q[work_i], v[work_i]))
        work.start()
        workers.append(work)

    qsize_new = 0
    qsize_old = 0

    bar = tqdm.tqdm(total=qsize_all)
    
    while (qsize_all - qsize_new) > 0:
        qsize_new = sum([i.value for i in v])
        bar.update(qsize_new - qsize_old)
        qsize_old = qsize_new
        time.sleep(0.01)
    
    print("sleep")

    [i.join() for i in workers]


