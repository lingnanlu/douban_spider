# coding=utf-8

from surprise import KNNBasic
from surprise import evaluate
from surprise import Dataset
from surprise import PredictionImpossible
from surprise import print_perf
from surprise import Reader
from surprise import dump
import pandas as pd
import numpy as np

class MyOwnKNN(KNNBasic):

    def __init__(self, **kwargs):
        KNNBasic.__init__(self, **kwargs)

    # # 评论指数计算方法
    # def _comment_indictor(self, x):
    #
    #     return (10 * x[2] + x[1]) / (x[0] + x[1] + x[2])
    #
    # # 日期分布指数
    # def _date_indictor(self, x):
    #
    #     return np.std(x)

    # 计算电影的评分统计信息
    def compute_movie_rating_stats(self, raw_rating_df, trainset):

        movie_rating_stats_df = raw_rating_df.groupby(['movieid', 'rating']).size().unstack().fillna(0)

        return movie_rating_stats_df

    # 计算电影的评论统计信息
    def compute_movie_comment_stats(self, raw_rating_df, trainset):

        movie_comment_stats_df = raw_rating_df.groupby(['movieid', 'comment_level']).size().unstack().fillna(0)


        return movie_comment_stats_df

    # 由电影评分统计信息和评论统计信息得出电影的B格表
    def compute_movie_bg(self, movie_rating_stats_df, movie_comment_stats_df):
        movie_rating_stats_df.columns = ['rating_0', 'rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5']
        movie_comment_stats_df.columns = ['comment_0', 'comment_1', 'comment_2']

        movie_rating_comment_stats_df = pd.merge(movie_rating_stats_df, movie_comment_stats_df, left_index=True,
                                              right_index=True)

        # 计算电影B格的具体方法
        def _compute_movie_func(x):
            rating_bg = (x[4] + x[5]) / (x[0] + x[1] + x[2] + x[3] + x[4] + x[5])
            comment_bg = x[-1] / (x[-1] + x[-2] + x[-3])
            return rating_bg + comment_bg

        return movie_rating_comment_stats_df.apply(_compute_movie_func, axis=1)

    # 由电影的B格表得出用户的B格表，用户所看B格电影越多，说明其B格越高
    def compute_user_bg(self, movie_bg_df, raw_rating_df, trainset):

        '''电影B格表和用户B格表用来对最终的预测评分进行修正，高B格用户对高B格电影评分更高'''

        user_rating_df_with_movie_bg = pd.merge(raw_rating_df, movie_bg_df.to_frame('bg'), left_on='movieid', right_index=True)

        # 暂时以用户所看电影B格的平均值作为用户的B格
        return user_rating_df_with_movie_bg.groupby(user_rating_df_with_movie_bg['userid'])['bg'].mean()

    def train(self, trainset):
        KNNBasic.train(self, trainset)


        raw_rating_df = pd.read_csv('new_ratings_all.txt',
                                    sep=':',
                                    header=None,
                                    names=['userid', 'movieid', 'rating', 'date', 'comment_level'],
                                    dtype={'movieid':np.str, 'userid':np.str})

        # self.comment_stats = raw_rating_df.groupby(['userid', 'comment_level']).size().unstack()
        # self.comment_stats.fillna(0, inplace=True)
        # self.comment_stats = self.comment_stats.apply(self._comment_indictor, axis=1)
        #
        # self.date_stats = raw_rating_df.groupby(['userid', 'date']).size()
        # self.date_stats = \
        #     self.date_stats.groupby(level='userid').apply(self._date_indictor)

        # 对相似度矩阵进行修正
        self.movie_bg_dg = self.compute_movie_bg(
            self.compute_movie_rating_stats(raw_rating_df, self.trainset),
            self.compute_movie_comment_stats(raw_rating_df, self.trainset)
        )

        self.user_bg_dg = self.compute_user_bg(
            self.movie_bg_dg, raw_rating_df, self.trainset
        )


    # 给定用户u，和电影i，得到一个数表明用户B格与电影B格的接进程度
    def bg_modify_factor(self, u, i):

        try:
            ruid = self.trainset.to_raw_uid(u)
        except ValueError:
            ruid = u.replace('UKN__', '')

        try:
            riid = self.trainset.to_raw_iid(i)
        except ValueError:
            print(i)
            riid = i.replace('UKN__', '')


        u_bg = self.user_bg_dg.ix[ruid]
        i_bg = self.movie_bg_dg.ix[riid]

        candinate = [-0.5, -0.4, -0.3, -0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
        return candinate[np.random.randint(0, len(candinate))]


    def estimate(self, u, i):

        details = {}
        # 基于Bg的评分
        est = 10 * self.bg_modify_factor(u, i)
        actual_k = 0
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            # 如果训练集中没有该用户或商品， 就设置为平均分
            est += self.trainset.global_mean
            details['was_impossible'] = True
            details['reason'] = 'User and/or item is unkown'
        else:
            x, y = self.switch(u, i)

            neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]

            # sort neighbors by similarity
            neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)

            # compute weighted average
            sum_sim = sum_ratings = 0
            for (_, sim, r) in neighbors[:self.k]:
                if sim > 0:
                    sum_sim += sim
                    sum_ratings += sim * r
                    actual_k += 1

            if actual_k < self.min_k:
                print('not enough neighbors')
                est += self.trainset.global_mean
                details['was_impossible'] = True
                details['reason'] = 'Not enough neighbors'
                # raise PredictionImpossible('Not enough neighbors.')
            else:
                est += sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details

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




