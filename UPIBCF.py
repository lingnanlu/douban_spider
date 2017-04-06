# coding=utf-8

from surprise import evaluate
from surprise import Dataset
from surprise import AlgoBase
from surprise import PredictionImpossible
from surprise import print_perf
from surprise import Reader
from surprise import dump
from scipy import stats
import pandas as pd
import numpy as np
import math


class UPIBCF(AlgoBase):

    def __init__(self, k=40, min_k=1, raw_rating_file='new_ratings_all.txt', sim_options={}, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k
        self.raw_rating_file = raw_rating_file

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        self.sim = self.compute_similarities()

        raw_rating_df = pd.read_csv(self.raw_rating_file,
                                sep=':',
                                header=None,
                                names=['uid', 'iid', 'rating', 'date', 'comment_type'],
                                dtype={'uid': np.str, 'iid': np.str})

        self.UCI = self.compute_UCI(raw_rating_df)
        self.URDI = self.compute_URDI(raw_rating_df)
        self.UPI = self.compute_UPI(self.UCI, self.URDI)

        self.iuid_2_UPI = {iuid: self.UPI_from(iuid) for iuid in trainset.all_users()}


    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            # 如果训练集中没有该用户或商品， 在父类中Catch该异常，设置平均分
            raise PredictionImpossible('User and/or item is unknown')

        est, details,  = self.estimate_by_UPIBCF(i, u)
        return est, details

    # 这里是具体融合用户专业度后的推荐算法，所以融合的方法在这里定义
    def estimate_by_UPIBCF(self, i, u):
        est1, details1 = self.estimate_by_traditional_CF(i, u)
        est2, details2 = self.estimate_by_UPI(i, u)
        est = 0.5 * est1 + 0.5 * est2
        return est, details1

    def UPI_from(self, iuid):

        return self.UPI['std'][self.trainset.to_raw_uid(iuid)]

    # user comment index
    @classmethod
    def compute_UCI(self, raw_rating_df):

        RNMUS = self.compute_RNMUS(raw_rating_df)
        IHot = self.compute_IHot(raw_rating_df)
        IHot = IHot.to_frame('hot')

        temp = pd.merge(raw_rating_df, IHot, left_on='iid', right_index=True)
        # comment_stats = raw_rating_df.groupby(['uid', 'comment_type']).size().unstack().fillna(0)

        # comment_stats['sum'] = comment_stats[0] + comment_stats[1] + comment_stats[2]

        # 评论指数计算方法
        def f(df):
            df['hot_weight'] = 1 / np.log(math.e + df['hot'] - 1)
            return np.sum(df['comment_type'] * df['hot_weight']) / np.sum(df['hot_weight'])

        temp = temp.groupby('uid').apply(f)
        temp = pd.concat([RNMUS, temp], axis=1)

        return 0.3* temp[0] + 0.7 * temp[1]

    # 用户所看电影数相对与看电影最多用户的比值，比值越来，说明用户所看电影越多
    @classmethod
    def compute_RNMUS(self, raw_rating_df):

        temp = raw_rating_df.groupby('uid').size()

        return temp / np.max(temp)

    # user rating date index, 实际就是变异系数
    @classmethod
    def compute_URDI(self, raw_rating_df):

        rating_date_stats = raw_rating_df.groupby(['uid', 'date']).size()

        # 计算均值
        mean = rating_date_stats.groupby(level='uid').apply(np.mean)

        rating_date_stats_withou_2 = rating_date_stats[rating_date_stats.index.get_level_values(1) != '1800-01-01']

        # 去除1800后，计算方差
        std = rating_date_stats_withou_2.groupby(level='uid').apply(np.std)
        # def f(s):
        #     s = s[s.index != '1800-01-01']
        #     return np.std(s)
        #
        # user_rating_date_stats_dict = {}
        # for userid, data_stats in rating_date_stats.groupby(level='uid'):
        #     user_rating_date_stats_dict[userid] = [
        #         np.sum(data_stats),
        #         f(data_stats)
        #     ]

        # return \
        #     pd.DataFrame.\
        #     from_dict(user_rating_date_stats_dict, orient='index').\
        #     rename(columns={0:'sum', 1:'std'})

        return std / mean

    # user profession index
    @classmethod
    def compute_UPI(self, UCI, URDI):

        # return pd.\
        #     concat([UCI,URDI['sum'],URDI['std']],axis=1).\
        #     rename(columns={0: 'UCI'})

        # 0.5后期可以调整

        uci_urdi = pd.concat([UCI, URDI], axis=1)
        return 0.5 * UCI + 0.5 / URDI

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

    def estimate_by_UPI(self, i, u):

        prof_user_rates = [(v, self.iuid_2_UPI[v], r) for (v, r) in self.trainset.ir[i]]
        prof_user_rates = sorted(prof_user_rates, key=lambda x: x[1], reverse=True)

        sum_prof = sum_ratings = actual_k = 0
        for (_, prof, r) in prof_user_rates[:self.k]:
            if prof > 0:
                sum_prof += prof
                sum_ratings += prof * r
                actual_k += 1
        est = sum_ratings / sum_prof
        details = {'actual_k': actual_k}
        return est, details

    # 用户被打分越多，说明越热门，在计算UCI中，权重越低
    @classmethod
    def compute_IHot(self, raw_rating_df):

        return raw_rating_df.groupby('iid').size()


if __name__ == '__main__':

    reader = Reader(line_format='user item rating', sep=':')

    data = Dataset.load_from_file('new_usr_ratings.txt', reader=reader)

    data.split(n_folds=2)

    algo = UPIBCF()

    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

else:
    pass




