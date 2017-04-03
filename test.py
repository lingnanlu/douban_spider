from surprise import Reader
from surprise import KNNBasic
from surprise import evaluate, print_perf
from surprise import Dataset
import numpy as np

# file_path = './usr_rating.txt'
# reader = Reader(line_format='user item rating', sep=':', rating_scale=(0, 5))
# data = Dataset.load_from_file(file_path, reader=reader)
# data.split(n_folds=3)
# algo = KNNBasic(sim_options={'name':'cosine', 'user_based':False})
# perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
# print_perf(perf)

np.zeros((41000, 41000), np.double)