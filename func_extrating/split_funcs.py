import os 
import re
import tqdm
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_cve_dir(root_path, cve_list):
    PATH = os.listdir(root_path)
    for path in PATH:
            cve_list.append(os.path.join(root_path, path))
    return cve_list

def main():
    err_dir = open('err_dir_list.txt', 'w')
    for i in tqdm.tqdm(range(1139)): 
        cve_list = []
        cve_list = get_cve_dir('/home/liucy/func_extracting/cve_data/' + str(i).zfill(3), cve_list) 
        for cv in cve_list:
            with open('runner','r' ) as fp:
                new_r = re.sub(r'/home/liucy/func_extracting/source_code \./input', cv + ' ./input', fp.read())
                with open('runner_0', 'w') as fp0:
                    fp0.write(new_r)
            subprocess.check_output('sh runner_0', shell=True)

            dif = os.path.join(cv, 'diff_file.txt')
            with open(dif, 'r') as fp:
                diff_file = fp.read()
                p1 = re.search('\n\n\ncommit\n', diff_file)
                p2 = re.search('(\npatch\n)', diff_file)
                if p1 == None or p2 == None:
                    err_dir.write(str(i).zfill(3))
                    continue
                cmt = diff_file[p1.span()[1]:p2.span()[0]-3]
                cve_id = cv.split('/')[-1]
                try:
                    subprocess.check_output('python extract_func_code.py --c ' + cmt + ' --d ' + str(i).zfill(3) + ' --i ' + cve_id, shell=True)
                except:
                    err_dir.write(str(i).zfill(3) +  '/' + cve_id + '\n')
                    pass
    err_dir.close()

if __name__ == '__main__':
    main()