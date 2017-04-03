import requests
from requests import RequestException
import re
import bs4
import logging as log
import sys
import numpy as np
import random
import string
import os
from bs4 import BeautifulSoup
from time import sleep
from time import gmtime, strftime

def fetch_movie_info(movie_id_list, save_file='movie_info.txt'):
    file = open(save_file, 'a')
    for movie_id in movie_id_list:
        url = 'https://movie.douban.com/subject/' + movie_id

        resp = requests.get(url)
        if resp.status_code != 200:
            log.info('http resp status:' + str(resp.status_code))
            raise Exception('Network Error')
        soup = BeautifulSoup(resp.text, 'html5lib')



