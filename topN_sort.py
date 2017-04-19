# coding=utf-8
from base import base


class topN_sort(base):
    def __init__(self, alpha=0.5, k=40, min_k=1, sim_options={}, **kwargs):
        base.__init__(self, alpha, k, min_k, sim_options, **kwargs)

    def train(self, trainset):
        base.train(self, trainset)

    def recommend(self, u, k, nitem):

        rank1 = self.recommend_by_rating_matrix(u, k, nitem)
        rank2 = self.recommend_by_behavior_matrix(u, k, nitem)

        rank = rank1 + rank2

        rank = sorted(rank, key=lambda x:x[1], reverse=True)

        #来自两处的推荐列表可能包括相同的item,所以需要去重
        rank = [item for item, rating in rank]

        return list(set(rank))[0:nitem]



