import subprocess
import os
import tqdm
import time
import eventlet
import argparse
from joern.all import JoernSteps
# from py2neo.packages.httpstream import http
# http.socket_timeout = 9999
from multiprocessing import Pool


def getAllFunctionNode(db):
    query_str = "queryNodeIndex('type:Function')"
    results = db.runGremlinQuery(query_str)
    return results

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        '--g',
                        type=str,
                        choices=['ast', 'cfg_dfg', 'ddg_cdg', 'cfg', 'cdg'],
                        help='graph type',
                        )
    return parser.parse_args()

def getFuncFile(db, func_id):
    query_str = "g.v(%d).in('IS_FILE_OF').filepath" % func_id
    re = db.runGremlinQuery(query_str)
    return re[0]

def excute_cmd(cmd):
    sh = '%s && %s' % (cmd[0], cmd[1])
    idx = cmd[1].find('-o ')
    tag = True
    svg_path = '/home/liucy/big_vul/cfg_dfg/' + cmd[1][idx+3:-1]
    if os.path.exists(svg_path):
        tag = False
    if tag == True:        
        eventlet.monkey_patch()
        try:
            with eventlet.Timeout(25,False):
                time.sleep(0.01)
                subprocess.check_output(sh, shell=True)
            # subprocess.check_output(sh, shell=True)
        except Exception as e:
            pass

if  __name__  == '__main__':
    j = JoernSteps()
    j.connectToDatabase()
    node_list = getAllFunctionNode(j)
    args = get_parameter()  # indicate graph type
    cmd = []
    # count = 0  # count the err files
    # err_file = open('./error_files.txt', 'w')
    print('[+] graph type: %s' % args.g)

    print('[+] get func files name from graph nodes...')
    bar = tqdm.tqdm(node_list)
    for node in bar:
        func_file = getFuncFile(j, node._id)
        func_file = func_file.split('/')[-1]
        func_file = func_file.split('.')[0] + '@' + node.properties['name']
        if args.g == 'ast':
            cmd1 = "echo 'getFunctionsByName(\"%s\").id' | joern-lookup -g | tail -n 1 | joern-plot-ast > %s_ast.dot" % (node.properties['name'], func_file)
        elif args.g == 'cfg_dfg':
            cmd1 = "echo 'getFunctionsByName(\"%s\").id' | joern-lookup -g | tail -n 1 | joern-plot-proggraph -cfg -dfg > %s.dot" % (node.properties['name'], func_file)
        elif args.g == 'ddg_cdg':
            cmd1 = "echo 'getFunctionsByName(\"%s\").id' | joern-lookup -g | tail -n 1 | joern-plot-proggraph -ddg -cdg > %s.dot" % (node.properties['name'], func_file)
        elif args.g == 'cfg':
            cmd1 = "echo 'getFunctionsByName(\"%s\").id' | joern-lookup -g | tail -n 1 | joern-plot-proggraph -cfg > %s.dot" % (node.properties['name'], func_file)
        elif args.g == 'cdg':
             cmd1 = "echo 'getFunctionsByName(\"%s\").id' | joern-lookup -g | tail -n 1 | joern-plot-proggraph -cdg > %s.dot" % (node.properties['name'], func_file)
        if args.g == 'ast':
            cmd2 = "dot -Tsvg %s_ast.dot -o %s_ast.svg;" % (func_file, func_file)
        else:
            cmd2 = "dot -Tsvg %s.dot -o %s.svg;" % (func_file, func_file)
        cmd.append([cmd1, cmd2])
        # print(cmd1, cmd2)
        bar.update()
    bar.close()

    p  = Pool(3)
    print('[+] get dot files and svg files...')
    pbar = tqdm.tqdm(cmd)
    for i in p.imap(excute_cmd, cmd):
        pbar.update()
    # for item in pbar:
    #     excute_cmd(item)
    #     pbar.update()
    # pbar.close()
