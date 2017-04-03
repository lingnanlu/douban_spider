# _*_ coding: utf-8 _*_

"""
test_demos.py by xianhu
"""

import re
import spider
import pymysql
import random
import string
import logging
import requests
from bs4 import BeautifulSoup
from demos_doubanmovies import MovieFetcher, MovieParser

def get_douban_movies():

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
        "Host": "movie.douban.com",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, sdch, br",
        "Accept-Language": "zh-CN, zh; q=0.8, en; q=0.6",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cookie" : 'bid=TWn93lyonNk; ll="118254"; gr_user_id=118696be-aa6a-42e9-a20f-932c29fcddac; viewed="5333562_5948760_4736118_4241826_1495763_1433583_2124114_6430747_24335672"; ps=y; _pk_ref.100001.8cb4=%5B%22%22%2C%22%22%2C1490076711%2C%22https%3A%2F%2Fmovie.douban.com%2Fsubject%2F1292052%2Freviews%22%5D; _ga=GA1.2.1671303578.1469101452; ue="lingnanlu@gmail.com"; dbcl2="33045345:gXYCq8g9sy4"; ck=5VGo; __utmt=1; _vwo_uuid_v2=98306AEEC1B83E40741FF0A8A58DC180|c5bbf2b10ddb9854ac614269b546a464; ap=1; push_noty_num=0; push_doumail_num=0; _pk_id.100001.8cb4=88a4be0bc4943075.1469262289.53.1490077859.1490064764.; _pk_ses.100001.8cb4=*; __utma=30149280.1671303578.1469101452.1490062608.1490076712.73; __utmb=30149280.16.10.1490076712; __utmc=30149280; __utmz=30149280.1489996683.69.35.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); __utmv=30149280.3304'
    }

    # 获取初始url
    all_urls = set()

    resp = requests.get("https://movie.douban.com/tag/", headers=headers, verify=False)
    assert resp.status_code == 200, resp.status_code

    soup = BeautifulSoup(resp.text, "html5lib")
    a_list = soup.find_all("a", href=re.compile(r"^/tag/", flags=re.IGNORECASE))
    all_urls.update([(a_soup.get_text(), "https://movie.douban.com" + a_soup.get("href")) for a_soup in a_list])

    # resp = requests.get("https://movie.douban.com/tag/?view=cloud", headers=headers, verify=False)
    # assert resp.status_code == 200, resp.status_code

    # soup = BeautifulSoup(resp.text, "html5lib")
    # a_list = soup.find_all("a", href=re.compile(r"^/tag/", flags=re.IGNORECASE))
    # all_urls.update([(a_soup.get_text(), "https://movie.douban.com" + a_soup.get("href")) for a_soup in a_list])
    logging.warning("all urls: %s", len(all_urls))

    # 构造爬虫
    dou_spider = spider.WebSpider(MovieFetcher(), MovieParser(max_deep=-1), spider.Saver(), spider.UrlFilter())
    for tag, url in all_urls:
        print(tag + ":" + url)
        dou_spider.set_start_url(url, ("index", tag), priority=1)
    dou_spider.start_work_and_wait_done(fetcher_num=20)
    return

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s\t%(levelname)s\t%(message)s")
    get_douban_movies()
