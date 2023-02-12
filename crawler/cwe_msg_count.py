import os
import tqdm
import operator
import json

if __name__  == '__main__':
    msg_root = './cwe_msg.txt'
    msg_data = open(msg_root, 'r').read()
    cwe_id = {}
    mul_cnt = 0  # 统计一个c文件含有多种漏洞的情况
    mul_cwe_file = open('multi_cwe_files.txt', 'w')
    msg_data = msg_data.split('-----------------------------------\n')
    for cwe_msg in tqdm.tqdm(msg_data):
        cwe_msg = cwe_msg.split('\n')
        cnt = 0
        cve_id = ''
        for cwe_line in cwe_msg:
            if cwe_line.startswith('***'):
                cve_id = cwe_line.replace('***', '')
            if cwe_line.startswith('CWE') or cwe_line.startswith('NVD'):
                cv = cwe_line.split('@')[0]
                cnt += 1
                if cv not in cwe_id:
                    cwe_id[cv] = 1
                else:
                    cwe_id[cv] += 1
        if cnt >= 2:
            mul_cnt += 1
            mul_cwe_file.write(cve_id + '\n')
    
    cwe_id = dict(sorted(cwe_id.items(), key=operator.itemgetter(1), reverse=True))
    with open('cwe_id_count.json', 'w') as fp:
        json.dump(cwe_id, fp)
    
    mul_cwe_file.close()
    print(len(cwe_id))
    print(mul_cnt)