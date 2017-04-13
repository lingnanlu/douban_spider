# coding=utf-8
from UBCFBase import UBCFBase
from surprise import Reader
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import PredictionImpossible
import numpy as np

class FS_Threshold(UBCFBase):

    def __init__(self, k=40, min_k=1, threshold=10,sim_options={}, **kwargs):
        UBCFBase.__init__(self, k, min_k, sim_options, **kwargs)
        self.threshold = threshold

    def train(self, trainset):

        UBCFBase.train(self, trainset)

        self.fusion_sim = self.get_fusion_sim(self.sim, self.behavior_sim)

    def estimate(self, u, i):
        # 利用融合相似度进行计算
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

    def get_fusion_sim(self, sim, behavior_sim):

        n_users = self.trainset.n_users
        placeholder = np.empty((n_users, n_users), dtype=bool)

        user_rating_items = {}

        for u in range(n_users):
            user_rating_items[u] = [item for item, rating in self.trainset.ur[u]]

        for i in range(n_users):
            for j in range(n_users):
                common_rate_count = len(np.intersect1d(user_rating_items[i], user_rating_items[j]))
                if common_rate_count > self.threshold:
                    placeholder[i, j] = True
                else:
                    placeholder[i, j] = False

        return np.where(placeholder, sim, behavior_sim)


if __name__ == '__main__':

    reader = Reader(line_format='user item rating', sep=':')
    data = Dataset.load_from_file('new_usr_ratings.txt', reader=reader)

    data.split(n_folds=3)

    algo = FS_Threshold()

    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

    print_perf(perf)

else:
    pass


