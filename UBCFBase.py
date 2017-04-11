# coding=utf-8

from surprise import AlgoBase
from surprise import PredictionImpossible
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import numpy as np
import math

class UBCFBase(AlgoBase):

    def __init__(self, assist_rating_file, alpha=0.5, k=40, min_k=1, sim_options={}, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k
        self.alpha = alpha
        self.assist_rating_df = pd.read_csv(
                                    assist_rating_file,
                                    sep=':',
                                    header=None,
                                    names=['uid', 'iid', 'rating', 'date', 'comment_type'],
                                    dtype={'uid': np.str, 'iid': np.str}
        )

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        self.trainset_assist_rating_df = self.get_trainset_assist_rating_df()
        self.UCI = self.compute_UCI()
        self.URDI = self.compute_URDI()
        self.sim = self.compute_similarities()
        self.behavior_sim = self.compute_behavior_similarities()

    def compute_behavior_similarities(self):

        print('Computing the behavior similarity matrix...')

        user_behavior_matrix = pd.concat([self.UCI, self.URDI], axis=1)
        user_behavior_sim_matrix = pairwise_distances(user_behavior_matrix, metric='euclidean')
        print('Done computing behavior similarity matrix')
        # return pd.DataFrame(user_behavior_sim_matrix, index=user_behavior_matrix.index, columns=user_behavior_matrix.index)
        # 该矩阵目前是科学计数法表示，没有index和column
        return user_behavior_sim_matrix


    # user comment index
    def compute_UCI(self):

        # 用户所看电影数相对与看电影最多用户的比值，比值越来，说明用户所看电影越多
        def compute_RNMUS(rating_df):
            temp = rating_df.groupby('uid').size()

            return temp / np.max(temp)

        def compute_IHot(rating_df):
            return rating_df.groupby('iid').size()

        RNMUS = compute_RNMUS(self.trainset_assist_rating_df)
        IHot = compute_IHot(self.trainset_assist_rating_df)
        IHot = IHot.to_frame('hot')
        #
        temp = pd.merge(self.trainset_assist_rating_df, IHot, left_on='iid', right_index=True)
        #
        # 评论指数计算方法
        def f(df):
            df['hot_weight'] = 1 / np.log(math.e + df['hot'] - 1)
            # return np.sum(df['comment_type'] * df['hot_weight']) / np.sum(df['hot_weight'])
            temp = df.groupby('comment_type')['hot_weight'].sum()

            if 1 in temp.index:
                return temp[1] / len(df)
            else:
                return 0

        temp = temp.groupby('uid').apply(f)
        temp = pd.concat([RNMUS, temp], axis=1)

        return self.alpha * temp[0] + (1 - self.alpha) * temp[1]

        # # uci定义一
        # def uci_1():
        #     temp = self.trainset_assist_rating_df.groupby(['uid', 'comment_type']).size().unstack().fillna(0)
        #
        #     return  (temp[1])/(temp[0] + temp[1])
        #
        # return uci_1()

    # user rating date index, 实际就是变异系数
    def compute_URDI(self):

        rating_date_stats = self.trainset_assist_rating_df.groupby(['uid', 'date']).size()

        # 计算均值
        mean = rating_date_stats.groupby(level='uid').apply(np.mean)

        rating_date_stats_withou_2 = rating_date_stats[rating_date_stats.index.get_level_values(1) != '1800-01-01']

        # 去除1800后，计算方差
        std = rating_date_stats_withou_2.groupby(level='uid').apply(np.std)

        return std / mean

    def estimate_by_cf(self, i, u):

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

    # 获得训练集所对应的原始数据集
    def get_trainset_assist_rating_df(self):

        trainset_users = [self.trainset.to_raw_uid(iuid) for iuid in self.trainset.all_users()]
        trainset_assist_rating_df = self.assist_rating_df[self.assist_rating_df['uid'].isin(trainset_users)]

        trainset_assist_rating_df.uid = trainset_assist_rating_df.uid.map(self.trainset.to_inner_uid)

        return trainset_assist_rating_df








