# coding=utf-8
from UBCFBase import UBCFBase
from surprise import Dataset
from surprise import Reader
from surprise import PredictionImpossible
from surprise import print_perf, evaluate
class GFusion(UBCFBase):

    def __init__(self, file='new_ratings_all.txt', alpha=0.5, k=40, min_k=1, sim_options={}, **kwargs):
        UBCFBase.__init__(self, file, alpha, k, min_k, sim_options, **kwargs)

    def train(self, trainset):

        UBCFBase.train(self, trainset)

        # #将新的维度添加到评分矩阵中去
        list_0 = []
        list_1 = []
        for u in self.trainset.all_users():
            list_0.append((u, self.user_behavior_matrix.ix[u][0]))
            list_1.append((u, self.user_behavior_matrix.ix[u][1]))

        self.trainset.ir[self.trainset.n_items] = list_0
        self.trainset.ir[self.trainset.n_items + 1] = list_1
        self.g_sim = self.compute_similarities()

    def estimate(self, u, i):
        # 利用融合相似度进行计算
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        neighbors = [(v, self.sim[u, v], r) for (v, r) in self.trainset.ir[i]]
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
    trainsets = []
    for trainset, testset in data.folds():
        trainsets.append(trainset)

    first_trainset = trainsets[0]

    algo = GFusion()

    pref = evaluate(algo, data, ['rmse'])

    print_perf(pref)
else:
    pass

