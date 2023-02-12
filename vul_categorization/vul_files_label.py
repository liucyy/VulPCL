import os
import tqdm
import json
import re
import operator
import argparse


def get_files_to_vul_data():
    func_label = {}
    with open('linux_labels.txt') as fp:
        data_lines = fp.readlines()
        for line in data_lines:
            line = line.replace('\n','')
            fn = line.split('@@')[0]
            la = line.split('@@')[-1]
            func_label[fn] = la

    g_type = ['ast', 'cfg_dfg', 'ddg_cdg']
    for gt in g_type:
        graph_files = os.listdir('./data/graph/linux/' + gt)
        for gf in tqdm.tqdm(graph_files):
            fn = gf.split('@')[0] + '@' + gf.split('@')[1] + '.c'
            if func_label[fn] == '1':
                s_vul_file_path = os.path.join('./data/graph/linux', gt, gf)
                d_vul_file_path = os.path.join('./vul_data/graph/linux', gt, gf)
                with open(s_vul_file_path, 'r', encoding='utf-8') as fp:
                    svg_data = fp.read()
                    with open(d_vul_file_path, 'w', encoding='utf-8') as fp1:
                        fp1.write(svg_data)

def get_valid_files_count():
    file_path = './vul_data/graph/linux/cfg_dfg'
    cwes_path = 'cwe_to_source_files.json'
    all_files = os.listdir(file_path)
    all_c_files = []
    for file in all_files:
        file = file.split('@')[0] + '@' +file.split('@')[1] + '.c'
        all_c_files.append(file)

    # print(len(all_c_files))
    cwe_to_file = json.load(open(cwes_path))
    valid_cf = {}
    valid_cf_count = {}
    for c, f in tqdm.tqdm(cwe_to_file.items()):
        valid_cf[c] = []
        valid_cf_count[c] = 0
        for fn in f:
            if fn in all_c_files:
                valid_cf[c].append(fn)
                valid_cf_count[c] += 1
    
    with open('valid_cwe_to_source_files.json', 'w') as fp:
        json.dump(valid_cf, fp)
    
    valid_cf_count = dict(sorted(valid_cf_count.items(), key=operator.itemgetter(1), reverse=True))
    with open('valid_cwe_to_source_files_count.json', 'w') as fp:
        json.dump(valid_cf_count, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cwe', type=str, default='CWE-416', help='input CWE id')
    args = parser.parse_args()

    '''统计各CWE对应的有效漏洞文件，部分在抽取cfg这些的过程中被遗弃了'''
    # get_valid_files_count()

    all_cwes = json.load(open('valid_cwe_to_source_files.json'))
    '''将漏洞文件存放到vul_data文件夹中'''
    # get_files_to_vul_data()

    # cwe_to_label = {'CWE-20': '1', 'CWE-416': '2', 'CWE-362': '3', 'CWE-200': '4', 'NVD-CWE-Other': '5', 'CWE-119': '6', 'CWE-264': '7', 'CWE-476': '8', 'NVD-CWE-noinfo': '9', 'CWE-400': '10'}
    cwe_to_label = {'CWE-787': '1', 'CWE-125': '2', 'CWE-20': '3', 'CWE-416': '4', 'CWE-190': '5', 'CWE-287': '6', 'CWE-476': '7', 'CWE-119': '8', 'CWE-862': '9', 'CWE-276': '10', 'CWE-200': '11'}
    '''对指定CWE id对应的文件标记为1，其余标记为0'''
    # with open('linux_' + args.cwe + '_labels.txt', 'w') as fp:
    #     multi_vul_files = []
    #     for c, f in all_cwes.items():
    #         if c == args.cwe:
    #             for fn in f:
    #                 multi_vul_files.append(fn)  ## 将一个文件归为一种漏洞后避免进行二次归类
    #                 fp.write(fn + '@@1\n')
    #         else:
    #             for fn in f:
    #                 if fn in multi_vul_files:
    #                     continue
    #                 fp.write(fn + '@@0\n')

    '''作为11分类问题，多出的1类为top-10之外的其他类行'''
    with open('linux_' + args.cwe + '_labels.txt', 'w') as fp:
        multi_vul_files = []
        for c, f in all_cwes.items():
            # if c == 'd1e7fd6462ca9fc76650fbe6ca800e35b24267da@linux_15021.c' or c == 'd1e7fd6462ca9fc76650fbe6ca800e35b24267da@linux_14535.c':
                # continue
            if args.cwe == 'CWE-TOP':
                if c in cwe_to_label:
                    for fn in f:
                        # print(fn)
                        multi_vul_files.append(fn)  
                        fp.write(fn + '@@' + cwe_to_label[c] + '\n')
                else:
                    for fn in f:
                        # if fn in multi_vul_files:
                        #     continue
                        fp.write(fn + '@@0\n')