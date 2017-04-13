# coding=utf-8
from UBCFBase import UBCFBase
from surprise import PredictionImpossible
from surprise import Reader
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import KNNBasic
from surprise import GridSearch
from time import gmtime, strftime
import numpy as np
import pandas as pd

class FS_Weight(UBCFBase):
    def __init__(self, k=40, min_k=1, alpha=0.5, beta=0.5, sim_options={}, **kwargs):
        UBCFBase.__init__(self, alpha, k, min_k, sim_options, **kwargs)
        self.beta = beta

    def train(self, trainset):
        UBCFBase.train(self, trainset)

        self.fusion_sim = self.get_fusion_sim(self.sim, self.behavior_sim)

    def get_fusion_sim(self, sim, behavior_sim):

        return self.beta * sim + (1 - self.beta) * behavior_sim

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        neighbors = [(v, self.fusion_sim[u, v], r) for (v, r) in self.trainset.ir[i]]
        # sort neighbors by fusion similarity
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)
        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (_, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1
        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')
        est = sum_ratings / sum_sim
        details = {'actual_k': actual_k}
        return est, details

if __name__ == '__main__':

    reader = Reader(line_format='user item rating', sep=':')
    data = Dataset.load_from_file('new_usr_ratings.txt', reader=reader)

    data.split(n_folds=3)

    # param_grid = {'alpha' : np.arange(0, 1, 0.1)}
   # param_grid = {'beta' : [0.5], 'alpha':[0, 1]}
    param_grid = {'alpha' : np.arange(0, 1.1, 0.1), 'beta' : np.arange(0, 1.1, 0.1)}
    grid_search = GridSearch(FS_Weight, param_grid, measures=['RMSE', 'MAE'])

    grid_search.evaluate(data)

    print(grid_search.best_score['RMSE'])
    print(grid_search.best_params['RMSE'])

    # print(grid_search.best_score['FCP'])
    # print(grid_search.best_params['FCP'])

    print(grid_search.best_score['MAE'])
    print(grid_search.best_params['MAE'])

    result_df = pd.DataFrame.from_dict(grid_search.cv_results)
    print(result_df)

    result_df.to_csv('FS_Weight_' + strftime('%m-%d-%H-%M', gmtime()))


    # algo1 = FS_Weight(beta=1)
    # algo2 = KNNBasic()
    #
    # for trainset, testset in data.folds():
    #     algo1.train(trainset)
    #     algo2.train(trainset)
    #
    #     predictions2 = algo2.test(testset)
    #     predictions1 = algo1.test(testset)
    #
    #     print(predictions2[:10])
    #     print(predictions1[:10])

else:
    pass