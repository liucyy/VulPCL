import json
import os
import operator
import tqdm


if __name__ == '__main__':
    '''获取文件漏洞信息'''
    file_labels = {}
    with open('linux_labels.txt', 'r') as fp:
        label_lines = fp.readlines()
        for line in label_lines:
            line = line.replace('\n', '')
            file_name = line.split('@@')[0]
            la = line.split('@@')[-1]
            file_labels[file_name] = la
    
    '''获取所有CWE及其包含的CVE编号'''
    cwe_cve = {}
    cwe_ccode = {}  # 获取每个cwe编号对应的漏洞文件
    with open('cwe_msg.txt', 'r') as fp:
        msg_data = fp.read().split('-----------------------------------\n')
        for msg in msg_data:
            msg_lines = msg.split('\n')
            for line in msg_lines:
                if line.startswith('***'):
                    cve_id = line.replace('***','')
                elif line.startswith('CWE') or line.startswith('NVD'):
                    cv = line.split('@')[0]
                    if cv not in cwe_cve:
                        cwe_cve[cv] = [cve_id]
                    else:
                        cwe_cve[cv].append(cve_id)
    
    '''获取各类CWE编号对应的漏洞文件及其数量'''    
    CWE_files = {}
    with open('linux_cve_labels.txt', 'r') as fp:
        fc = fp.read().split('\n')
        for line in tqdm.tqdm(fc):
            file_name = line.split('@@')[0]
            cve_id = line.split('@@')[-1]
            for k, v in cwe_cve.items():
                if cve_id in v:
                    if k not in cwe_ccode:
                        CWE_files[k] = 1
                        cwe_ccode[k] = [file_name]
                    else:
                        CWE_files[k] += 1
                        cwe_ccode[k].append(file_name)
    
    CWE_files = dict(sorted(CWE_files.items(), key=operator.itemgetter(1), reverse=True))
    print(len(cwe_ccode))
    with open('cwe_files_count.json', 'w') as fp:
        json.dump(CWE_files, fp)
        
    with open('cwe_to_source_files.json', 'w') as fp:
        json.dump(cwe_ccode, fp)