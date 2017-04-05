# coding=utf-8

from surprise import KNNBasic
from surprise import evaluate
from surprise import Dataset
from surprise import AlgoBase
from surprise import PredictionImpossible
from surprise import print_perf
from surprise import Reader
from surprise import dump
import pandas as pd
import numpy as np

class MyOwnKNN(AlgoBase):

    def __init__(self, k=40, min_k=1, sim_options={}, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k

    # 日期分布指数
    def _date_indictor(self, x):

        return np.std(x)

    # # 计算电影的评分统计信息
    # def compute_movie_rating_stats(self, raw_rating_df, trainset):
    #
    #     movie_rating_stats_df = raw_rating_df.groupby(['movieid', 'rating']).size().unstack().fillna(0)
    #
    #     return movie_rating_stats_df
    #
    # # 计算电影的评论统计信息
    # def compute_movie_comment_stats(self, raw_rating_df, trainset):
    #
    #     movie_comment_stats_df = raw_rating_df.groupby(['movieid', 'comment_level']).size().unstack().fillna(0)
    #
    #
    #     return movie_comment_stats_df
    #
    # # 由电影评分统计信息和评论统计信息得出电影的B格表
    # def compute_movie_bg(self, movie_rating_stats_df, movie_comment_stats_df):
    #     movie_rating_stats_df.columns = ['rating_0', 'rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5']
    #     movie_comment_stats_df.columns = ['comment_0', 'comment_1', 'comment_2']
    #
    #     movie_rating_comment_stats_df = pd.merge(movie_rating_stats_df, movie_comment_stats_df, left_index=True,
    #                                           right_index=True)
    #
    #     # 计算电影B格的具体方法
    #     def _compute_movie_func(x):
    #         rating_bg = (x[4] + x[5]) / (x[0] + x[1] + x[2] + x[3] + x[4] + x[5])
    #         comment_bg = x[-1] / (x[-1] + x[-2] + x[-3])
    #         return rating_bg + comment_bg
    #
    #     return movie_rating_comment_stats_df.apply(_compute_movie_func, axis=1)
    #
    # # 由电影的B格表得出用户的B格表，用户所看B格电影越多，说明其B格越高
    # def compute_user_bg(self, movie_bg_df, raw_rating_df, trainset):
    #
    #     '''电影B格表和用户B格表用来对最终的预测评分进行修正，高B格用户对高B格电影评分更高'''
    #
    #     user_rating_df_with_movie_bg = pd.merge(raw_rating_df, movie_bg_df.to_frame('bg'), left_on='movieid', right_index=True)
    #
    #     # 暂时以用户所看电影B格的平均值作为用户的B格
    #     return user_rating_df_with_movie_bg.groupby(user_rating_df_with_movie_bg['userid'])['bg'].mean()

    def train(self, trainset):

        AlgoBase.train(self, trainset)
        self.sim = self.compute_similarities()

        raw_rating_df = pd.read_csv('new_ratings_all.txt',
                                    sep=':',
                                    header=None,
                                    names=['userid', 'movieid', 'rating', 'date', 'comment_level'],
                                    dtype={'movieid':np.str, 'userid':np.str})

        self.user_comment_indictor = self.get_user_comment_indictor(raw_rating_df)

        self.user_rating_date_indictor = self.get_user_rating_date_indictor(raw_rating_df)

        self.user_professional_indictor = self.get_user_professional_indictor(self.user_comment_indictor, self.user_rating_date_indictor)


        self.inner_uid_2_professional = {iuid :self.user_professional_indictor['std'][self.trainset.to_raw_uid(iuid)] for iuid, _ in self.trainset.ur.iteritems()}

        # 对相似度矩阵进行修正
        # self.movie_bg_dg = self.compute_movie_bg(
        #     self.compute_movie_rating_stats(raw_rating_df, self.trainset),
        #     self.compute_movie_comment_stats(raw_rating_df, self.trainset)
        # )
        #
        # self.user_bg_dg = self.compute_user_bg(
        #     self.movie_bg_dg, raw_rating_df, self.trainset
        # )

    def get_user_rating_date_indictor(self, raw_rating_df):

        rating_date_stats = raw_rating_df.groupby(['userid', 'date']).size()

        def f(s):
            s = s[s.index != '1800-01-01']
            return np.std(s)

        user_rating_date_stats_dict = {}
        for userid, data_stats in rating_date_stats.groupby(level='userid'):
            user_rating_date_stats_dict[userid] = [
                np.sum(data_stats),
                f(data_stats)
            ]

        return pd.DataFrame.from_dict(user_rating_date_stats_dict, orient='index').rename(columns={0:'sum', 1:'std'})

    def get_user_comment_indictor(self, raw_rating_df):

        comment_stats = raw_rating_df.groupby(['userid', 'comment_level']).size().unstack().fillna(0)

        comment_stats['sum'] = comment_stats[0] + comment_stats[1] + comment_stats[2]

        # 评论指数计算方法
        def f(x):
            return (10 * x[2] + x[1]) / (x[0] + x[1] + x[2])

        return comment_stats.apply(f, axis=1)

    # 给定用户u，和电影i，得到一个数表明用户B格与电影B格的接进程度
    # def bg_modify_factor(self, u, i):
    #
    #     try:
    #         ruid = self.trainset.to_raw_uid(u)
    #     except ValueError:
    #         ruid = u.replace('UKN__', '')
    #
    #     try:
    #         riid = self.trainset.to_raw_iid(i)
    #     except ValueError:
    #         print(i)
    #         riid = i.replace('UKN__', '')
    #
    #
    #     u_bg = self.user_bg_dg.ix[ruid]
    #     i_bg = self.movie_bg_dg.ix[riid]
    #
    #     candinate = [-0.5, -0.4, -0.3, -0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    #     return candinate[np.random.randint(0, len(candinate))]

    def estimate(self, u, i):
        # details = {}
        # # 基于Bg的评分
        # est = 10 * self.bg_modify_factor(u, i)
        # actual_k = 0
        # if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
        #     # 如果训练集中没有该用户或商品， 就设置为平均分
        #     est += self.trainset.global_mean
        #     details['was_impossible'] = True
        #     details['reason'] = 'User and/or item is unkown'
        # else:
        #     x, y = self.switch(u, i)
        #
        #     neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        #
        #     # sort neighbors by similarity
        #     neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)
        #
        #     # compute weighted average
        #     sum_sim = sum_ratings = 0
        #     for (_, sim, r) in neighbors[:self.k]:
        #         if sim > 0:
        #             sum_sim += sim
        #             sum_ratings += sim * r
        #             actual_k += 1
        #
        #     if actual_k < self.min_k:
        #         print('not enough neighbors')
        #         est += self.trainset.global_mean
        #         details['was_impossible'] = True
        #         details['reason'] = 'Not enough neighbors'
        #         # raise PredictionImpossible('Not enough neighbors.')
        #     else:
        #         est += sum_ratings / sum_sim
        #
        # details = {'actual_k': actual_k}
        # return est, details

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            # 如果训练集中没有该用户或商品， 就设置为平均分, 在父类中Catch该异常，设置平均分
            raise PredictionImpossible('User and/or item is unknown')

        est1, details1 = self.compute_by_traditional_cf(i, u)
        est2, details2 = self.compute_by_professional(i, u)
        est = 0.5 * est1 + 0.5 * est2
        return est, details1

    def compute_by_traditional_cf(self, i, u):
        print('compute_by_traditional_cf')
        # 得到所以评价过商品i的用户
        neighbors = [(v, self.sim[u, v], r) for (v, r) in self.trainset.ir[i]]
        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)
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

    def get_professional(self, iuid):

        return self.user_professional_indictor['std'][self.trainset.to_raw_uid(iuid)]

    def compute_by_professional(self, i, u):

        print('compute_by_professional')
        users_rate_u = [
            (v, self.inner_uid_2_professional[v], r)
            for (v, r) in self.trainset.ir[i]]
        users_rate_u = sorted(users_rate_u, key=lambda tple: tple[1], reverse=True)

        sum_prof = sum_ratings = actual_k = 0
        for (_, prof, r) in users_rate_u[:self.k]:
            if prof > 0:
                sum_prof += prof
                sum_ratings += prof * r
                actual_k += 1
        est = sum_ratings / sum_prof
        details = {'actual_k': actual_k}
        return est, details

    def get_user_professional_indictor(self, user_comment_indictor, user_rating_date_indictor):

        return pd.concat(
            [user_comment_indictor,
             user_rating_date_indictor['sum'],
             user_rating_date_indictor['std']],
            axis=1
        ).rename(columns={0:'comment_indictor'})



if __name__ == '__main__':

    reader = Reader(line_format='user item rating', sep=':')

    data = Dataset.load_from_file('new_usr_ratings.txt', reader=reader)

    data.split(n_folds=3)

    algo = MyOwnKNN()


    # data.build_full_trainset()
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    # perf = evaluate(algo, data, measures=['RMSE', 'MAE'], with_dump=True, dump_dir='./dump/')
    # perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    #
    # print_perf(perf)
else:
    pass




