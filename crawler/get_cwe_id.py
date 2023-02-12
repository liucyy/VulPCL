# -*- coding: utf8 -*-
# SmartContactSpider.py
import requests
from bs4 import BeautifulSoup
import traceback
import os
import json
import tqdm
import time
import datetime
from sys import stdin
import xlwt

def printtime():
    print(time.strftime("%Y-%m-%d %H:%M:%S:", time.localtime()), end=' ')
    return 0


def gets_all_links(eachLine, fp, cve_id):
    '''伪装成浏览器'''
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36'}
    payload = {"id":"apk_name"}
    failedTimes = 100
    while True:  # 在制定次数内一直循环，直到访问站点成功
        if (failedTimes <= 0):
            printtime()
            print("失败次数过多，请检查网络环境！")
            break

        failedTimes -= 1
        try:
            # 以下except都是用来捕获当requests请求出现异常时，
            # 通过捕获然后等待网络情况的变化，以此来保护程序的不间断运行
            response = requests.get(eachLine, params=payload, headers=headers, timeout=5)  # 访问网站
            break

        except:
            time.sleep(1)
            pass

    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text, "html.parser")
    
    msg_total = soup.find_all('div', attrs={'data-testid':'vuln-technical-details-container'})
    # print(nameTotal)
    # item.SCName = nameTotal[0].text
    for list in msg_total:
        cwe_msg = list.find_all('td')
        fp.write('***' + cve_id + '***' + '\n')
        steps = int(len(cwe_msg)/3)  # 三个连在一起的是一个CWE id对应的描述信息
        for i in range(steps):
            fp.write(cwe_msg[i*3].text.replace('\n', '') + '@' + cwe_msg[i*3 + 1].text + '\n')
        fp.write('-----------------------------------\n')

    return 0

def getslinks():
    try:
        # 修改此处:请根据自己的需求改成自己想要的路径。
        SCAddress = open("./cwe_address_0.txt", "r").readlines()
    except:
        printtime()
        print('打开GitHub URL地址仓库错误！请检查文件目录是否正确！')

    # print(SCAddress)
    with open('cwe_msg_0.txt', 'w', encoding='utf-8') as fp:
        for eachLine in tqdm.tqdm(SCAddress):
            eachLine = eachLine.replace('\n', '')
            cve_id = eachLine.split('/')[-1]
            gets_all_links(eachLine, fp, cve_id)  # 获取github advisory database中相关链接的核心函数
            # break
    return 0

if __name__ == '__main__':
    '''将需要爬取的网页链接写入txt文件中'''
    # cve_data = open('loss_cve_id.txt').read().split('\n')
    # with open('cwe_address_0.txt', 'w') as fp:
    #     for v in cve_data:
    #         fp.write('https://nvd.nist.gov/vuln/detail/' + v + '\n')
    getslinks()