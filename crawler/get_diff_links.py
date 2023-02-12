# -*- coding: utf8 -*-
import requests
from bs4 import BeautifulSoup
import traceback
import os
import time
import random
import tqdm
import re
import datetime
from sys import stdin
from multiprocessing import Pool
import xlwt

def gets_link(addr):
    '''获取github advisory database中相关链接的核心函数'''
    addr = addr.split('@')[-1].replace('\n', '')

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
        # print(soup)
    
        link_list = soup.find_all('a', attrs={'rel':'nofollow'})
        cve_id = ''
        for lk in link_list:
            name = lk.text.strip()
            link = lk.get('href')
            if re.match('https://nvd.nist.gov/vuln/detail/CVE', name) != None:
                cve_id = re.sub('https://nvd.nist.gov/vuln/detail/', '', name)
            if re.match('https://git.kernel.org', link) != None or re.match('http://git.kernel.org', link) != None:
                result = cve_id + '@' + link + '\n'
                return result
    return addr + '\n'

fp = open('diff_links.txt', 'w', encoding='utf-8')
headers = {}
user_agent_list =  ["Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36", 
                    "Mozilla/5.0 (WindoWS NT 10.0;WOW64) AppleWebKit/537.36 (KHTML, 1ike Gecko) Chrome/67.0.3396.99 Safari/537.36",
                    "Mozilla/5.0 (Windows NT 10.0;…) Gecko/20100101 Firefox/61.0",
                    "Mozilla/5.0 (Windows NT 10.0;WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36",
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36(KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36",
                    "Mozilla/5.0 (Windows NT 6.1;WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36",
                    "Mozilla/4.0 (compatible; MSIE 7.0; WindowS NT 6.0)",
                    "Mozilla/5.0 (Macintosh ; U; PPC Mac 0SX10.5; en-US; rv:1.9.2.15) Gecko/20110303 Firefox/3.6.15",
                    ]
'''伪装成浏览器'''
headers["Connection"] = 'close'

if __name__ == '__main__':
    try:
        '''修改此处:请根据自己的需求改成自己想要的路径。'''
        all_addr = open('./data_links.txt', 'r', encoding='utf-8').readlines()
    except:
        print('打开GitHub URL地址仓库错误！请检查文件目录是否正确！')

    # p = Pool(2)
    result_list = []
    bar = tqdm.tqdm(all_addr)
    bar.set_description('crawling diff files links')
    # for i in p.imap(gets_link, all_addr):
    #     result_list.append(i)
    #     bar.update()
    # bar.close()
    # with open('diff_links.txt', 'w', encoding='utf-8') as fp:
    #     for i in result_list:
    #         fp.write(i)
    with open('diff_links.txt', 'w', encoding='utf-8') as fp:
        for addr in bar:
            result = gets_link(addr)
            fp.write(result)
            bar.update()
        bar.close()