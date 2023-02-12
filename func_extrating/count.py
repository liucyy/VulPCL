import os
import tqdm
import re

def get_diff(root_path, diff_list):
    PATH = os.listdir(root_path)
    for path in PATH:
        tmp = os.path.join(root_path, path)
        if os.path.isdir(tmp):
            get_diff(tmp, diff_list)
        else:
            diff_list.append(tmp)
    return diff_list


if __name__ == '__main__':
    '''汇总cve_data下的函数信息和相应的CVE id、commit id以及函数名'''
    root_path = './cve_data/'
    cnt = 0
    with open('func_to_file.txt', 'w') as fp:
        for i in tqdm.tqdm(range(1139)):
            dir = root_path + str(i).zfill(3)
            list = []
            list = get_diff(dir, list)
            for l in list:
                if l.endswith('.c') and re.search(r'\\funcs\\', l):
                    tag = True  # 删除空文件的标记
                    with open(l, 'r', encoding='utf-8') as fp1:
                        try:
                            func = fp1.readlines()[0].replace('\n', '')
                        except:
                            tag = False
                            pass
                    if tag == False:
                        os.remove(l)
                        continue
                    file = l.split('\\')[-3] + '@' + l.split('\\')[-1]
                    fp.write(func + '@' + file + '\n')
                    cnt += 1
    print(cnt)