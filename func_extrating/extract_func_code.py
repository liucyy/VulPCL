from ast import arg
import os
import re
from xml.dom.minidom import parse
import tqdm
import argparse

def get_file_path(root_path, file_list):
    PATH = os.listdir(root_path)
    for path in PATH:
        co_path = os.path.join(root_path, path)
        if os.path.isfile(co_path):
            file_list.append(co_path)
        elif os.path.isdir(co_path):
            get_file_path(co_path, file_list)
    return file_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='0d3d077cd4f1154e63a9858e47fe3fb1ad0c03e5', help='input commit id')
    parser.add_argument('--d', type=str, default='000', help='input dir name')
    parser.add_argument('--i', type=str, default='cve-2005-4886', help='input cve id')
    args = parser.parse_args()

    root_path = './input'
    file_path = []
    file_path = get_file_path(root_path, file_path) 
    print('[+] extracting and save functions from code...')
    for file in file_path:
        cnt = 0
        with open(file, 'r') as fp:
            scode = fp.read()
            fn = scode.split('\n')[0].split('"')[1].split('/')[-1].split('.')[0]
            scode = re.sub(r'<source.*>\n', '', scode)
            scode = re.sub(r'</source>\n', '&@#@$', scode)
            all_funcs = scode.split('&@#@$')
            for func in all_funcs[:-1]:
                func_name = './cve_data/' + args.d + '/' + args.i + '/funcs/'+ args.c + '@' + fn + '_' + str(cnt) + '.c'
                cnt += 1
                with open(func_name, 'w') as fp1:
                    fp1.write(func)
