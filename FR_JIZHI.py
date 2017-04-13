# coding=utf-8
from UBCFBase import UBCFBase
from surprise import PredictionImpossible
from surprise import Reader
from surprise import Dataset
from surprise import evaluate, print_perf
import math
import numpy as np

class FR_JIZHI(UBCFBase):
    def __init__(self, k=40, min_k=1, alpha=0.5, file='new_ratings_all.txt', sim_options={}, **kwargs):
        UBCFBase.__init__(self, file, alpha, k, min_k, sim_options, **kwargs)

    def train(self, trainset):
        UBCFBase.train(self, trainset)

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        est_by_rating_cf, details = UBCFBase.estimate_by_cf(self, u, i)
        est_by_behavior_cf = self.estimate_by_behavior_cf(u, i)

        if est_by_behavior_cf > est_by_rating_cf:
            return est_by_behavior_cf
        else:
            return est_by_rating_cf

    def estimate_by_behavior_cf(self, u, i):

        neighbors = [(v, self.behavior_sim[u, v], r) for (v, r) in self.trainset.ir[i]]
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
        return est

if __name__ == '__main__':

    reader = Reader(line_format='user item rating', sep=':')
    data = Dataset.load_from_file('new_usr_ratings.txt', reader=reader)

    data.split(n_folds=3)

    algo = FR_JIZHI()

    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

    print_perf(perf)

else:
    pass