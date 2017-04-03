import requests
from requests import RequestException
from requests import adapters

def get_proxy():
    return requests.get("http://192.168.202.129:5000/get/").content.decode('UTF-8')

def delete_proxy(proxy):
    requests.get("http://192.168.202.129/delete/?proxy={}".format(proxy))

# your spider code

def spider():
    # ....
    resp = requests.get('https://www.douban.com', proxies={"http": "http://{}".format(get_proxy())})
    print(resp.status_code)
    # ....

proxies = get_proxy()
url = 'https://www.douban.com/'
proxy = get_proxy()
while True:
    try:
        #print(proxy)
        resp = requests.get(url, proxies={'https' : 'https://' + '124.248.215.75:80'})
        print(resp.status_code)
    except Exception as e:
        print(e)
        #delete_proxy(proxy)
        # proxy = get_proxy()
# proxies = {
#     'http': 'http://' + proxy,
#     'https' : 'https://' + proxy
# }

# def fetch_url(url, proxies):
#
#
#     while True:
#         try:
#             print(proxies)
#             resp = requests.get(url, proxies=proxies, timeout=5, verify=False)
#             print(url + "resp:" + str(resp.status_code) + "  proxy:" + proxies)
#             if resp.status_code == 200:
#                 return resp
#             else:
#                 print(resp.status_code)
#                 delete_proxy(proxies['http'])
#                 proxy = get_proxy()
#                 proxies['http'] = 'http://' + proxy
#                 proxies['https'] = 'https://' + proxy
#         except adapters.ProxyError as e:
#             print(e)
#             delete_proxy(proxies['http'])
#             proxy = get_proxy()
#             proxies['http'] = 'http://' + proxy
#             proxies['https'] = 'https://' + proxy
#         except adapters.ConnectTimeoutError as e:
#             print(e)
#             delete_proxy(proxies['http'])
#             proxy = get_proxy()
#             proxies['http'] = 'http://' + proxy
#             proxies['https'] = 'https://' + proxy
#
# url = 'https://www.douban.com/'
# for i in range(0, 50000):
#     print(i)
#     fetch_url(url, proxies)

