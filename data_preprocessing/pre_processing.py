import pickle
import os
import re
import pandas as pd
import json
import tqdm

def get_file_path(root_path, file_list):
    PATH = os.listdir(root_path)
    for path in PATH:
        # print(path)
        co_path = os.path.join(root_path, path)
        if os.path.isfile(co_path):
            file_list.append(co_path)
        elif os.path.isdir(co_path):
            get_file_path(co_path, file_list)
    return file_list

def get_uc_filtering(code):
    '''过滤掉中文字符'''
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    code = re.sub(pattern, '', code)
    return code

def code_filtering(file_list, flag):
    '''去除code中的注释、中文'''
    count = 0  # 统计有问题代码数量
    new_file_list = []
    pbar = tqdm.tqdm(total=len(file_list))
    pbar.set_description('comments filtering')
    for file in file_list:
        tag = False
        '''注意windows目录下的文件路径格式'''
        file_name = file.split('\\')[-1]
        with open(file, 'r') as fp:
            if flag == True:
                new_file = './data/code_after_filtering/FFmpeg/' + file_name
            else:
                new_file = './data/code_after_filtering/qemu/' + file_name
            new_file_list.append(new_file)
            source_code = fp.read().split('\n')
            with open(new_file, 'w') as fp1:
                for code_line in source_code:
                    if tag == False:
                        code_line = re.sub(r'//[\s|\S]*', '', code_line)
                        code_line = re.sub(r'/\*.*\*/', '', code_line)
                        if re.match(r'.*/\*.*', code_line) != None:
                            code_line = re.sub(r'/\*.*', '', code_line)
                            tag =True
                    else:
                        if re.match(r'.*\*/', code_line) != None:
                            code_line = re.sub(r'.*\*/', '', code_line)
                            tag = False
                        else:
                            line = re.sub(r'.*', '', code_line)
                    fp1.write(code_line + '\n')
            fp1.close()
            pbar.update()
    pbar.close()

    bar = tqdm.tqdm(new_file_list)
    bar.set_description('unChinese filtering')
    for file in bar:
        try:
            with open(file, 'rb') as fp:
                code = fp.read()
                code = code.decode('utf-8', 'ignore')
                code = get_uc_filtering(code)
        except Exception as e:
            count += 1
            print(file)
        bar.update()
    bar.close()
    print(count)
    return new_file_list

def all_files_to_pkl(file_list, labels_path, flag):
    pathes = []
    codes = []
    labels = []
    file_labels = {}
    print('code labels path:' + labels_path)
    with open(labels_path, 'r') as fp:
        all_lines = fp.read().split('\n')
        for line in all_lines:
            line = line.split('@@')
            file_labels[line[0][:-2]] = line[-1]

    pbar = tqdm.tqdm(file_list)
    pbar.set_description('generate code_after_filtering.pkl')
    for path_list in pbar:
        path = path_list.split('/')
        pathes.append(path[-1])
        codes.append(open(path_list, 'r').read())
        labels.append(file_labels[path[-1][:-2]])
        pbar.update()
    pbar.close()
    code_pkl = {'ids': pathes, 'codes': codes, 'labels': labels}
    df = pd.DataFrame(code_pkl)
    if flag == True:
        df.to_pickle('FFmpeg_files.pkl')
    else:
        df.to_pickle('qemu_files.pkl')


if __name__ == '__main__':
    root_path = './data/source_code/'
    # source_file = json.load(open('../data/function.json'))
    # code_label = open('./code_labels.txt', 'w')
    # pbar = tqdm.tqdm(enumerate(source_file))
    # pbar.set_description('json to code')
    # for i, item in pbar:
    #     filename = item['commit_id'] + '@' + item['project'] + '_' + str(i) + '.c'
    #     fp = open('./data/source_code/' + filename, 'w')
    #     fp.write(item['func'])
    #     code_label.write(filename + '@@' + str(item['target']) + '\n')
    #     pbar.update()
    # pbar.close()
    # code_label.close()

    source_path= './function.json'
    
    source_file = json.load(open(source_path))
    root_path = './data/source_code'
    print('[+] json to code files...')
    bar = tqdm.tqdm(total=len(source_file))
    FFmpeg_labels = open('./FFmpeg_labels.txt', 'w')
    qemu_labels = open('./qemu_labels.txt', 'w')
    for i, item in enumerate(source_file):
        # sub_dir = str(i).zfill(5)
        pro_path = os.path.join(root_path, item['project'])
        file_name = item['commit_id'] + '@' +  item['project'] + '_' + str(i) +'.c'
        if item['project'] == 'FFmpeg':
            # dir_path = os.path.join(pro_path, sub_dir)
            # os.mkdir(dir_path)
            with open(os.path.join(pro_path, file_name), 'w') as fp:
                fp.write(item['func'])
            fp.close()
            FFmpeg_labels.write(file_name + '@@' + str(item['target']) + '\n')
        elif item['project'] == 'qemu':
            # dir_path = os.path.join(pro_path, sub_dir)
            # os.mkdir(dir_path)
            with open(os.path.join(pro_path, file_name), 'w') as fp:
                fp.write(item['func'])
            fp.close()
            qemu_labels.write(file_name + '@@' + str(item['target']) + '\n')
        bar.update()
    bar.close()
    FFmpeg_labels.close()
    qemu_labels.close()

    ffmpeg_file_list = []
    qemu_file_list = []
    ffmpeg_labels_path = './FFmpeg_labels.txt'
    qemu_labels_path = './qemu_labels.txt'
    # '''获取文件路径'''
    get_file_path('./data/source_code/FFmpeg', ffmpeg_file_list)
    get_file_path('./data/source_code/qemu', qemu_file_list)
    '''去除源代码中的注释和中文'''
    print('[+] code filtering...')
    ffmpeg_file_list = code_filtering(ffmpeg_file_list, flag=True)
    qemu_file_list = code_filtering(qemu_file_list, flag=False)

    '''生成整体代码的pkl文件'''
    print('[+] get all_files.pkl...')
    all_files_to_pkl(ffmpeg_file_list, ffmpeg_labels_path, flag=True)
    all_files_to_pkl(qemu_file_list, qemu_labels_path, flag=False)
