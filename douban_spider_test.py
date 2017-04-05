# coding=utf-8
import unittest
from douban_spider import *

class TestSpider(unittest.TestCase):


    USR_URLS_FILE = 'test_usr_urls.txt'
    USR_RATING_FILE = 'test_usr_ratings.txt'

    def setUp(self):
        # log.disable(log.INFO)


        pass
    #####################################################################################
    # 测试用户数不足页数
    def test_parse_user_indexs_url_0(self):

        movie_url = 'https://movie.douban.com/subject/26331917'    # 碟仙诡谭2

        user_urls = parse_user_indexs_url(movie_url, page=10, file=TestSpider.USR_URLS_FILE)

        self.assertEqual(len(user_urls), 69)

    # 测试用户数足页数
    def test_parse_user_indexs_url_1(self):

        movie_url = 'https://movie.douban.com/subject/25900945'    # 美女与野兽

        user_urls = parse_user_indexs_url(movie_url, page=10, file=TestSpider.USR_URLS_FILE)

        self.assertEqual(len(user_urls), 200)

    #####################################################################################
    # 测试用户少于一页观看电影数
    def test_parse_user_movie_rating_from_collection0(self):

        user_movie_collection_url = 'https://movie.douban.com/people/159392315/collect'

        ratings = parse_user_movie_rating_from_collection(user_movie_collection_url, 1, file=TestSpider.USR_RATING_FILE)

        self.assertEqual(len(ratings), 1)

    # 测试用户不足页数的观看电影数
    def test_parse_user_movie_rating_from_collection1(self):
        user_movie_collection_url = 'https://movie.douban.com/people/156573791/collect'

        ratings = parse_user_movie_rating_from_collection(user_movie_collection_url, 4, file=TestSpider.USR_RATING_FILE)

        self.assertEqual(len(ratings), 9)

    # 测试用户足页数的观看电影数
    def test_parse_user_movie_rating_from_collection2(self):
        user_movie_collection_url = 'https://movie.douban.com/people/34347136/collect'

        ratings = parse_user_movie_rating_from_collection(user_movie_collection_url, 5, file=TestSpider.USR_RATING_FILE)

        self.assertEqual(len(ratings), 75)

    # 测试用户所有观看电影数
    def test_parse_user_movie_rating_from_collection3(self):
        user_movie_collection_url = 'https://movie.douban.com/people/loseyoursef/collect'

        ratings = parse_user_movie_rating_from_collection(user_movie_collection_url, file=TestSpider.USR_RATING_FILE)

        self.assertEqual(len(ratings), 115)


    #####################################################################################
    # 测试用户少于一页评论电影数
    def test_parse_user_movie_rating_from_reviews0(self):

        user_movie_reviews_url = 'https://movie.douban.com/people/156573791/reviews'

        ratings = parse_user_movie_rating_from_reviews(user_movie_reviews_url, 1, file=TestSpider.USR_RATING_FILE)

        self.assertEqual(len(ratings), 1)

    # 测试用户不足页数的评论电影数
    def test_parse_user_movie_rating_from_reviews1(self):

        user_movie_reviews_url = 'https://movie.douban.com/people/34347136/reviews'

        ratings = parse_user_movie_rating_from_reviews(user_movie_reviews_url, 10, file=TestSpider.USR_RATING_FILE)

        self.assertEqual(len(ratings), 32)

    # 测试用户足页数的评论电影数
    def test_parse_user_movie_rating_from_reviews2(self):

        user_movie_reviews_url = 'https://movie.douban.com/people/34347136/reviews'

        ratings = parse_user_movie_rating_from_reviews(user_movie_reviews_url, 2, file=TestSpider.USR_RATING_FILE)

        self.assertEqual(len(ratings), 20)

    # 测试用户所有评论电影数
    def test_parse_user_movie_rating_from_reviews3(self):

        user_movie_reviews_url = 'https://movie.douban.com/people/34347136/reviews'

        ratings = parse_user_movie_rating_from_reviews(user_movie_reviews_url, file=TestSpider.USR_RATING_FILE)

        self.assertEqual(len(ratings), 32)

    # 测试用户没有评论电影数
    def test_parse_user_movie_rating_from_reviews4(self):
        user_movie_reviews_url = 'https://movie.douban.com/people/loseyoursef/reviews'

        ratings = parse_user_movie_rating_from_reviews(user_movie_reviews_url, 1, file=TestSpider.USR_RATING_FILE)

        self.assertEqual(len(ratings), 0)


    #####################################################################################


