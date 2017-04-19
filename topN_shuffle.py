# coding=utf-8
from base import base
import random

class topN_shuffle(base):
    def __init__(self, alpha=0.5, k=40, min_k=1, sim_options={}, **kwargs):
        base.__init__(self, alpha, k, min_k, sim_options, **kwargs)

    def train(self, trainset):
        base.train(self, trainset)

    def recommend(self, u, k, nitem):

        rank1 = self.recommend_by_rating_matrix(u, k, nitem)
        rank2 = self.recommend_by_behavior_matrix(u, k, nitem)

        rank1 = [item for item, rating in rank1]
        rank2 = [item for item, rating in rank2]

        # 去重
        rank = set()
        rank.update(rank1)
        rank.update(rank2)

        rank = list(rank)
        random.shuffle(rank)


        return rank[0:nitem]





