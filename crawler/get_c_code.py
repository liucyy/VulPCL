# -*- coding: utf8 -*-
import requests
from bs4 import BeautifulSoup
import traceback
import os
import time
import random
import tqdm
import pandas as pd
import pickle as pkl
import re
import datetime
from sys import stdin
from multiprocessing import Pool
import xlwt
                   
def gets_link(addr):
    '''获取c源代码文件'''
    cve_id = addr.split('@')[0]
    c_id = addr.split('@')[1]
    addr = addr.split('@')[-1].replace('\n','')
    tag = False
    failedTimes = 100
    while True:                 # 在制定次数内一直循环，直到访问站点成功
        if (failedTimes <= 0):  # print("失败次数过多，请检查网络环境！")
            break

        failedTimes -= 1
        try:
            headers["User-Agent"] = random.choice(user_agent_list)
            response = requests.get(addr, headers=headers)  # 访问网站
            tag = True
            break
        except:
            time.sleep(3)
            pass
    
    if tag == True:
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "html.parser")
        # print(soup.get_text())

        with open('./cve_files/' + cve_id + '/' + c_id, 'w', encoding='utf-8') as fp:
            fp.write(soup.get_text())

headers = {}
'''伪装成浏览器'''
user_agent_list =  ["Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36", 
                    "Mozilla/5.0 (WindoWS NT 10.0;WOW64) AppleWebKit/537.36 (KHTML, 1ike Gecko) Chrome/67.0.3396.99 Safari/537.36",
                    "Mozilla/5.0 (Windows NT 10.0;…) Gecko/20100101 Firefox/61.0",
                    "Mozilla/5.0 (Windows NT 10.0;WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36",
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36(KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36",
                    "Mozilla/5.0 (Windows NT 6.1;WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36",
                    "Mozilla/4.0 (compatible; MSIE 7.0; WindowS NT 6.0)",
                    "Mozilla/5.0 (Macintosh ; U; PPC Mac 0SX10.5; en-US; rv:1.9.2.15) Gecko/20110303 Firefox/3.6.15",
                    ]
headers["Connection"] = 'close'

if __name__ == '__main__':
    try:
        '''修改此处:请根据自己的需求改成自己想要的路径。'''
        all_addr = open('./c_links.txt', 'r', encoding='utf-8').readlines()
    except:
        print('打开GitHub URL地址仓库错误！请检查文件目录是否正确！')

    # p = Pool(2)
    print('[+] crawling diff files and c file links...')
    bar = tqdm.tqdm(all_addr)
    for addr in bar:
        gets_link(addr)
        bar.update()
    bar.close()