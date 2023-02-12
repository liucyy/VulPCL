import os
import re

if __name__ == '__main__':
    cnt = 0
    with open('diff_links.txt', 'r') as fp:
        links = fp.readlines()
        for lk in links:
            if lk.startswith('CVE') and re.search('patch', lk) == None:
                cnt += 1
    print(cnt)