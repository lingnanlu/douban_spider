# coding=utf-8
import requests
import re
import bs4
import logging as log
import sys
import numpy as np
from bs4 import BeautifulSoup
from time import sleep
from time import gmtime, strftime
import pandas as pd

FORMAT = '[%(asctime)s]-[%(funcName)s:%(lineno)d]-[%(message)s]'
current_time = strftime('%m-%d-%H-%M', gmtime())
log.basicConfig(format=FORMAT, level=log.INFO, handlers=[log.StreamHandler(sys.stdout),log.FileHandler('log_real_' + current_time + '.txt')])

headers = {
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36 Edge/14.14393',
    'Accept': 'text/html, application/xhtml+xml, image/jxr, */*',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN',
    'Connection': 'Keep-Alive',
    'Host': 'movie.douban.com',
    'Referer': 'https//www.douban.com'
}

def build_cookie_from_str(cookie_string):
    key_values = cookie_string.split(';')
    cookies = {}
    for i in key_values:
        cookies[i.split('=')[0].strip()] = i.split('=')[1].strip()

    return cookies

cookies = build_cookie_from_str(
    'll=118254; bid=92coublApbM; dbcl2=33045345:wE42wpgswVY; ck=C4pK; push_noty_num=0; push_doumail_num=0; _pk_id.100001.4cf6=b142c5d91fb7cd4c.1491039994.2.1491043543.1491039994.; _pk_ses.100001.4cf6=*')

proxies = {
    'http': 'socks5://127.0.0.1:1080',
    'https': 'socks5://127.0.0.1:1080'
}


# 利用电影来得到用户，因为每部电影只能得到最多200位为其打分的用户，所以使用多部电影得到用户的并集
def parse_user_indexs_url(movie_url, page=10, file='usr_urls_' + current_time + '.txt'):
    '''
    根据电影URL得到前200个评价用户的主页URL
    :param movie_url:
    :return:
    '''

    if movie_url[-1] != '/':
        movie_url = movie_url + '/collections?start='
    else:
        movie_url = movie_url + 'collections?start='

    _file = open(file, 'a')
    user_indexs = set()
    for start in range(0, 20 * page, 20):
        url = movie_url + str(start)
        log.info("processing " + url)
        resp = fetch_url(url)
        if resp.status_code != 200:
            log.info('http resp status:' + str(resp.status_code))
            raise Exception('Network Error')
        soup = BeautifulSoup(resp.text, 'html5lib')
        user_list_tag = soup.find('div', class_='sub_ins')

        user_tags = set()

        user_tags.update(user_list_tag.find_all('a'))
        if len(user_tags) == 0:
            break

        user_list = [user['href'] for user in user_tags]

        for item in user_list:
            _file.write("%s\n" % item)

        user_indexs.update(user_list)

        sleep_random_time()

    _file.close()
    return user_indexs


def get_user_movie_collection_url(user_index_url):
    '''
    example :

    input : https://www.douban.com/people/59641110/
    output : https://movie.douban.com/people/59641110/collect
    :param user_index_url:
    :return:
    '''
    return user_index_url.replace('www', 'movie') + 'collect'


def get_user_movie_reviews_url(user_index_url):
    '''
    input : https://www.douban.com/people/59641110/
    output : https://movie.douban.com/people/59641110/reviews
    :param user_urls:
    :return:
    '''
    return user_index_url.replace('www', 'movie') + 'reviews'


def get_userid(user_index_url):
    if user_index_url[-1] == '/':
        return user_index_url.split('/')[-2]
    else:
        return user_index_url.split('/')[-1]


def parse_user_movie_rating(user_index_url, collection_page=-1, review_page=-1):
    log.info('user_index_url ' + user_index_url)
    user_movie_reviews_url = get_user_movie_reviews_url(user_index_url)
    user_movie_collection_url = get_user_movie_collection_url(user_index_url)

    movie_rating = []

    movie_rating += parse_user_movie_rating_from_collection(user_movie_collection_url, collection_page)
    movie_rating += parse_user_movie_rating_from_reviews(user_movie_reviews_url, review_page)

    return movie_rating


def parse_user_movie_rating_from_reviews(user_movie_reviews_url, page=-1, file='usr_rating_' + current_time + '.txt'):
    log.info('processing ' + user_movie_reviews_url)
    resp = fetch_url(user_movie_reviews_url)

    user_id = user_movie_reviews_url.split('/')[-2]

    if resp.status_code != 200:
        log.info('http resp status:' + str(resp.status_code))
        raise Exception('Network Error')

    try:
        _file = open(file, 'a')

        soup = BeautifulSoup(resp.text, 'html5lib')

        paginator = soup.find(class_='paginator')

        pages = []
        if paginator == None:
            # 说明用户没有评论电影或不满足一页
            pages = [0]
        else:
            pages = _parse_pages(paginator, page, step=10)

        movie_ratings = []
        for i in pages:
            url = user_movie_reviews_url + "?start=" + str(i)
            log.info('processing ' + url)

            resp = fetch_url(url)
            if resp.status_code != 200:
                log.info('http resp status:' + str(resp.status_code))
                raise Exception('Network Error')

            soup = BeautifulSoup(resp.text, 'html5lib')

            movies_section = soup.find('div', class_='article')

            movie_tags = movies_section.find_all('ul')

            if len(movie_tags) == 0:
                break

            for movie_tag in movie_tags:

                movieid = movie_tag.find('a')['href'].split('/')[-2]

                rating_tag = movie_tag.find(class_=re.compile('^allstar'))

                rating = 0
                if rating_tag != None:
                    rating = int(rating_tag['class'][0][7])

                # 这里如果要解析date得再发一遍http请求，而长评影片所占比例很少，得不偿失，所以设置一个特殊值，
                # 分析数据时，对其进行过滤即可
                date = '1800-01-01'

                comment_level = 2

                _file.write(
                    '%s\n' % (user_id + ':' + movieid + ':' + str(rating) + ':' + date + ':' + str(comment_level)))
                movie_ratings.append((movieid, rating, date, comment_level))

            sleep_random_time()
    finally:
        _file.close()

    return movie_ratings


def parse_user_movie_rating_from_collection(user_movie_collection_url, page=-1,
                                            file='usr_rating_' + current_time + '.txt'):
    '''
    :param user_movie_collection_url:
    :return: [(movieid, rating, date, comment_level)...]}
    '''

    log.info('processing ' + user_movie_collection_url)
    resp = fetch_url(user_movie_collection_url)

    if resp.status_code != 200:
        log.info('http resp status:' + str(resp.status_code))
        raise Exception('Network Error')

    user_id = user_movie_collection_url.split('/')[-2]

    try:
        _file = open(file, 'a')
        soup = BeautifulSoup(resp.text, 'html5lib')

        paginator = soup.find(class_='paginator')

        pages = []
        if paginator == None:
            # 说明用户没有评论电影或不满足一页
            pages = [0]
        else:
            pages = _parse_pages(paginator, page, step=15)

        movie_ratings = []
        for i in pages:
            url = user_movie_collection_url + "?start=" + str(i)

            log.info('processing ' + url)

            resp = fetch_url(url)
            if resp.status_code != 200:
                log.info('http resp status:' + str(resp.status_code))
                raise Exception('Network Error')

            soup = BeautifulSoup(resp.text, 'html5lib')

            movies_section = soup.find('div', class_='grid-view')

            if movies_section == None:
                continue

            movie_div_tags = [movie_div_tag for movie_div_tag in movies_section.children
                              if type(movie_div_tag) == bs4.element.Tag]

            current_page_ratings = _parse_movie_ratings(movie_div_tags)

            for movie_rating in current_page_ratings:
                line = user_id + ':' + movie_rating[0] + ':' + str(movie_rating[1]) + ':' + movie_rating[2] + ':' + str(
                    movie_rating[3])
                _file.write('%s\n' % line)

            movie_ratings += current_page_ratings

            sleep_random_time()
    finally:
        _file.close()
    return movie_ratings


def _parse_pages(paginator, page, step):
    max_pager = 0
    if paginator != None:
        max_pager = int(paginator.find_all('a')[-2].string)

    if page == -1:
        starts = range(0, max_pager * step, step)
    else:
        starts = range(0, min(max_pager, page) * step, step)

    return starts


# (movieid, rating, date, comment_level)
def _parse_movie_ratings(movie_div_tags):
    movie_rating = []

    for movie_div_tag in movie_div_tags:

        movieid = movie_div_tag.find('a')['href'].split('/')[-2]

        rating_tag = movie_div_tag.find(class_=re.compile('rating\d-t'))

        rating = 0
        if rating_tag != None:
            rating = int(rating_tag['class'][0][6])

        date = movie_div_tag.find(class_='date').string

        comment_tag = movie_div_tag.find(class_='comment')

        comment_level = 0

        if comment_tag != None:
            comment_level = 1

        movie_rating.append((movieid, rating, date, comment_level))

    return movie_rating


def fetch_url(url):
    return requests.get(url, headers=headers, cookies=cookies)


def sleep_random_time():
    # pass
    a = [1]
    random_time = a[np.random.randint(0, len(a))]
    sleep(random_time)


def fetch_usr_urls():
    movie_urls = [
        'https://movie.douban.com/subject/25900945',  # 美女与野兽
        'https://movie.douban.com/subject/25934014',  # 爱乐之城
        'https://movie.douban.com/subject/26354572',  # 欢乐好声音
        'https://movie.douban.com/subject/26331917',  # 碟仙诡谭2
        'https://movie.douban.com/subject/26961684',  # 玛格丽特的春天
        'https://movie.douban.com/subject/26862259',  # 乘风破浪
        'https://movie.douban.com/subject/2277018',  # 麦兜响当当
        'https://movie.douban.com/subject/26820836',  # 八月
        'https://movie.douban.com/subject/25986180',  # 釜山行
        'https://movie.douban.com/subject/26683290',  # 你的名字
    ]

    user_urls = set()
    for movie_url in movie_urls:
        user_urls.update(parse_user_indexs_url(movie_url))
    print(len(user_urls))


def modify_comment_2_movieid(review_ids, file='review_to_movie_' + current_time + '.txt'):
    with open(file, 'a') as _file:

        for review_id in review_ids:
            url = "https://movie.douban.com/review/" + review_id
            log.info("processing " + url)

            resp = fetch_url(url)

            if resp.status_code != 200:
                log.info('http resp status:' + str(resp.status_code))
                #raise Exception('Network Error')
                continue

            soup = BeautifulSoup(resp.text, 'html5lib')

            movie_id = soup.find(class_='main-hd').find_all('a')[-1]['href'].split('/')[-2]

            _file.write('%s\n' % (review_id + ':' + movie_id))

            sleep_random_time()


if __name__ == '__main__':

# user_urls_file = 'uniq_usr_urls.txt'
#
# with open(user_urls_file) as f:
#     user_urls = f.readlines()
#
#     user_urls = [item.strip() for item in user_urls]
#
#     for user_url in user_urls:
#         user_ratings = parse_user_movie_rating(user_url)
#         print('[user_url]' + '[' + user_url + ']' + ':' + str(len(user_ratings)))

    rating = pd.read_csv('usr_rating_all.txt', header=None, sep=':', names=['userid', 'movieid', 'rating', 'date', 'comment_level'], dtype={'userid':np.str, 'movieid':np.str})

    review_ids = rating[rating['comment_level'] == 2]['movieid'].tolist()[2762:]

    print(len(review_ids))

    modify_comment_2_movieid(review_ids)

else:
    pass
