# coding=utf-8
from base import base

class top_Fusion(base):

    def __init__(self, alpha=0.5, k=40, min_k=1, sim_options={}, **kwargs):
        base.__init__(self, alpha, k, min_k, sim_options, **kwargs)

    def train(self, trainset):

        base.train(self, trainset)

        # #将新的维度添加到评分矩阵中去
        list_0 = []
        list_1 = []
        for u in self.trainset.all_users():
            list_0.append((u, self.user_behavior_matrix.ix[u][0]))
            list_1.append((u, self.user_behavior_matrix.ix[u][1]))

        self.trainset.ir[self.trainset.n_items] = list_0
        self.trainset.ir[self.trainset.n_items + 1] = list_1

        # 全局相似度
        self.g_sim = self.compute_similarities()

    def recommend(self, u, k, nitem):

        rank = dict()

        u_rating_items = [item for item, rating in self.trainset.ur[u]]
        # 获得用户于其它所有用户的相似度
        neighbors = [item for item in enumerate(self.g_sim[u])]

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


