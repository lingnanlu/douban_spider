# coding=utf-8
from base import base
from surprise import PredictionImpossible
from surprise import Reader
from surprise import Dataset

class FS_Weight(base):
    def __init__(self, k=40, min_k=1, alpha=0.5, beta=0.5, sim_options={}, **kwargs):
        base.__init__(self, alpha, k, min_k, sim_options, **kwargs)
        self.beta = beta

    def train(self, trainset):
        base.train(self, trainset)

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

    def recommend(self, u, k, nitem):

        rank = dict()

        u_rating_items = [item for item, rating in self.trainset.ur[u]]
        # 获得用户于其它所有用户的相似度
        neighbors = [item for item in enumerate(self.fusion_sim[u])]

        # 获得K近邻
        k_neighbors = sorted(neighbors, key=lambda x:x[1], reverse=True)[0:k]

        for v, wuv in k_neighbors:
            for item, rating in self.trainset.ur[v]:
                if item in u_rating_items:
                    continue
                rank.setdefault(item, 0)
                rank[item] += wuv * rating


        rank = sorted(rank.items(), key=lambda x : x[1], reverse=True)[0:nitem]

        return [item for item, rating in rank]

if __name__ == '__main__':

    reader = Reader(line_format='user item rating', sep=':')
    train_file = 'new_usr_ratings.train'
    test_file = 'new_usr_ratings.test'
    data = Dataset.load_from_folds([(train_file, test_file)], reader)
    #
    # param_grid = {'beta' : [0, 0.3]}
    # grid_search = GridSearch(FS_Weight, param_grid, measures=['RMSE'])
    #
    # grid_search.evaluate(data)
    #
    # print(grid_search.best_score['RMSE'])
    # print(grid_search.best_params['RMSE'])
    #
    # result_df = pd.DataFrame.from_dict(grid_search.cv_results)
    # print(result_df)
    #
    # result_df.to_csv('FS_Weight_' + strftime('%m-%d-%H-%M', gmtime()))

    trainset, testset = data.folds().next()

    algo = FS_Weight(beta=1)
    algo.train(trainset)

    ks = [10,20]
    nitems = [10, 20]
    for k in ks:
        for nitem in nitems:
            algo.precisionAndRecall(testset, k, nitem)
            algo.coverage(k, nitem)

else:
    pass