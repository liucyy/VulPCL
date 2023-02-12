# -*- coding: utf8 -*-
# SmartContactSpider.py
import requests
from bs4 import BeautifulSoup
import traceback
import os
import time
import datetime
from sys import stdin
import xlwt

class SCItem(object):
    SCAddress = None    # 网页地址
    SCName = None    # 漏洞描述
    SCLink = None    # 漏洞链接

def piplines(SCItems):
    print('开始保存数据...')
    now = time.strftime('%Y-%m-%d', time.localtime())
    filePath = "D:\\master subject\\Bug Prediction\\Paper Code\\results"
    fileName = 'SC-' + now + '.xls'
    book = xlwt.Workbook(encoding = 'utf-8', style_compression = 0)
    sheet = book.add_sheet('SC')
    i = 0
    while i < len(SCItems):
        item = SCItems[i]
        sheet.write(i, 0, item.SCAddress)
        sheet.write(i, 1, item.SCName)
        sheet.write(i, 2, item.SCBalance)
        sheet.write(i, 3, item.SCCreator)
        sheet.write(i, 4, item.SCFirTxn)
        i = i + 1
    book.save(filePath + fileName)
    print('保存完成')

def printtime():
    print(time.strftime("%Y-%m-%d %H:%M:%S:", time.localtime()), end=' ')
    return 0


def gets_all_links(eachLine, fp):
    web_name = eachLine

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
            printtime()
            print('正在连接的的网址链接是 ' + eachLine, end='')
            response = requests.get(eachLine, params=payload, headers=headers, timeout=5)  # 访问网站
            break

        except requests.exceptions.ConnectionError:
            printtime()
            print('ConnectionError！请等待3秒！')
            time.sleep(3)

        except requests.exceptions.ChunkedEncodingError:
            printtime()
            print('ChunkedEncodingError！请等待3秒！')
            time.sleep(3)

        except:
            printtime()
            print('Unfortunitely,出现未知错误！请等待3秒！')
            time.sleep(3)

    response.encoding = response.apparent_encoding

    soup = BeautifulSoup(response.text, "html.parser")
    
    
    # item.SCAddress = web_name
    
    nameTotal = soup.find_all('a', attrs={'class':'Link--primary v-align-middle no-underline h4 js-navigation-open'})
    # print(nameTotal)
    # item.SCName = nameTotal[0].text
    for list in nameTotal:
        item = SCItem()
        item.SCAddress = web_name
        item.SCName = list.text.strip()
        item.SCLink = 'https://github.com' + list.get('href')
        fp.write(item.SCName + '@' + item.SCLink + '\n')
        # print(item.SCLink)

    printtime()
    print('link_files make done！')

    return 0

def getslinks():
    try:
        # 修改此处:请根据自己的需求改成自己想要的路径。
        SCAddress = open("D:\\master subject\\Bug Prediction\\Paper Code\\address.txt", "r").readlines()
    except:
        printtime()
        print('打开GitHub URL地址仓库错误！请检查文件目录是否正确！')

    # print(SCAddress)
    with open('data_links.txt', 'w', encoding='utf-8') as fp:
        for eachLine in SCAddress:
            eachLine = eachLine.replace('\n', '')
            gets_all_links(eachLine, fp)  # 获取github advisory database中相关链接的核心函数
            # break
    return 0

if __name__ == '__main__':
    '''将需要爬取的网页链接写入txt文件中'''
    # with open('address.txt', 'w') as fp:
    #     for i in range(305):
    #         website = 'https://github.com/advisories?page=' + str(i + 1) +'&query=Linux'
    #         fp.write(website + '\n')
    getslinks()