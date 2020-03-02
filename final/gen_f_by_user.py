import pandas as pd
import pickle

from tqdm.notebook  import tqdm
from sqlalchemy import create_engine

clients = pd.read_csv('data/clients.csv', index_col='client_id')

engine = create_engine(
        'postgres://postgres:postgres@localhost:5432/data',
        isolation_level="READ_UNCOMMITTED",
        pool_reset_on_return="commit"
    )

def gen_f(user):
    
    transactions = user.groupby('transaction_id')
    
    days = pd.to_datetime(user.transaction_datetime)
    
    new_f = {
        'client_id': user.iloc[0].client_id,
        'store_count': len(user.store_id.unique()),
        'product_type_count': len(user.product_id.unique()),
        'transaction_count': len(user.transaction_id.unique()),
        'days_sum': (days.max() - days.min()).days
    }
    
    days = pd.Series(days.unique())
    days = (days - days.shift(periods=1)).dropna()
    
    new_f['days_mean'] = days.mean().days
    new_f['days_std'] = days.std().days
    new_f['days_median'] = days.median().days
    
    for i in ['product_quantity', 'trn_sum_from_iss', 'trn_sum_from_red']:
        new_f[f"{i}_sum"] = user[i].sum()
        new_f[f"{i}_mean"] = user[i].mean()
        new_f[f"{i}_std"] = user[i].std()
        new_f[f"{i}_median"] = user[i].median()
    
    for i in ['regular_points_received', 'express_points_received', 'regular_points_spent', 'express_points_spent', 'purchase_sum']:
        new_f[f"{i}_sum"] = transactions[i].max().sum()
        new_f[f"{i}_mean"] = transactions[i].max().mean()
        new_f[f"{i}_std"] = transactions[i].max().std()
        new_f[f"{i}_median"] = transactions[i].max().median()
    
    return new_f
    
users = []
for i, client in enumerate(tqdm(clients.index)):
    user = pd.read_sql_query(f"select * from purchases where client_id = '{client}'", engine)
    users.append(gen_f(user))

with open('data/users.pkl', 'wb') as file:
    pickle.dump(users, file)