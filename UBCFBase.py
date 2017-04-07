# coding=utf-8

from surprise import AlgoBase
from surprise import PredictionImpossible
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import numpy as np
import math


class UBCFBase(AlgoBase):

    def __init__(self, k=40, min_k=1, raw_rating_file='new_ratings_all.txt', sim_options={}, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k
        self.raw_rating_file = raw_rating_file

        self.raw_rating_df = pd.read_csv(self.raw_rating_file,
                                    sep=':',
                                    header=None,
                                    names=['uid', 'iid', 'rating', 'date', 'comment_type'],
                                    dtype={'uid': np.str, 'iid': np.str})

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        self.sim = self.compute_similarities()
        self.behavier_sim = self.compute_behavier_similarities()

    def compute_behavier_similarities(self):

        self.UCI = self.compute_UCI(self.raw_rating_df)
        self.URDI = self.compute_URDI(self.raw_rating_df)

        user_behavior_matrix = pd.concat([self.UCI, self.URDI], axis=1)

        user_behavior_sim_matrix = pairwise_distances(user_behavior_matrix, metric='euclidean')

        return pd.DataFrame(user_behavior_sim_matrix, index=user_behavior_matrix.index, columns=user_behavior_matrix.index)

    # user comment index
    @classmethod
    def compute_UCI(self, raw_rating_df):

        # 用户所看电影数相对与看电影最多用户的比值，比值越来，说明用户所看电影越多
        def compute_RNMUS(raw_rating_df):
            temp = raw_rating_df.groupby('uid').size()

            return temp / np.max(temp)

        def compute_IHot(raw_rating_df):
            return raw_rating_df.groupby('iid').size()

        RNMUS = compute_RNMUS(raw_rating_df)
        IHot = compute_IHot(raw_rating_df)
        IHot = IHot.to_frame('hot')

        temp = pd.merge(raw_rating_df, IHot, left_on='iid', right_index=True)

        # 评论指数计算方法
        def f(df):
            df['hot_weight'] = 1 / np.log(math.e + df['hot'] - 1)
            return np.sum(df['comment_type'] * df['hot_weight']) / np.sum(df['hot_weight'])

        temp = temp.groupby('uid').apply(f)
        temp = pd.concat([RNMUS, temp], axis=1)

        return 0.3* temp[0] + 0.7 * temp[1]

    # user rating date index, 实际就是变异系数
    @classmethod
    def compute_URDI(self, raw_rating_df):

        rating_date_stats = raw_rating_df.groupby(['uid', 'date']).size()

        # 计算均值
        mean = rating_date_stats.groupby(level='uid').apply(np.mean)

        rating_date_stats_withou_2 = rating_date_stats[rating_date_stats.index.get_level_values(1) != '1800-01-01']

        # 去除1800后，计算方差
        std = rating_date_stats_withou_2.groupby(level='uid').apply(np.std)

        return std / mean


    def estimate_by_traditional_CF(self, i, u):

        # 得到所以评价过商品i的用户
        neighbors = [(v, self.sim[u, v], r) for (v, r) in self.trainset.ir[i]]
        # sort neighbors by similarity
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





