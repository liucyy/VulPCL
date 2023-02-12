import sys
import os
from collections import Counter
import tqdm
import json
import pickle as pkl
import argparse
import operator
import pandas as pd
import networkx as nx
from deepwalk_embed.deepwalk_embedding import deepwalk
import numpy as np
from xml.dom.minidom import parse
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
sys.path.append('.')


def get_svg_path(ast_root, ast_path_list):
    path_list = os.listdir(ast_root)
    for path in path_list:
        cur_path = os.path.join(ast_root, path)
        if os.path.isfile(cur_path) and cur_path.endswith('.svg'):
            ast_path_list.append(cur_path)
        elif os.path.isdir(cur_path):
            get_svg_path(cur_path, ast_path_list)
    return ast_path_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str, default='FFmpeg', help='input project name')
    args = parser.parse_args()

    '''选定的ddg_cdg中节点类型'''
    key_ntypes0 = ["ExpressionStatement", "Condition", "IdentifierDeclStatement", "Parameter", "BreakStatement", "ReturnStatement", 
                "GotoStatement", "ForInit", "IncDecOp", "ContinueStatement", "AssignmentExpr",  "ClassDefStatement", "Expression"]

    '''选定的ast中节点类型'''
    key_ntypes1 = ["Identifier", "Argument", "PrimaryExpression", "PtrMemberAccess", "ExpressionStatement", "AssignmentExpr", "CallExpression", 
                "ArgumentList", "ArrayIndexing", "Condition", "IdentifierDecl", "IdentifierDeclType", "UnaryOp", "UnaryOperator", 
                "IfStatement", "AdditiveExpression", "IdentifierDeclStatement", "RelationalExpression", "MultiplicativeExpression", "MemberAccess", 
                "Parameter", "ParameterType", "EqualityExpression", "ReturnStatement", "IncDec", "IncDecOp", "ForStatement", "ElseStatement", 
                "ShiftExpression", "ForInit", "AndExpression", "BitAndExpression", "BreakStatement", "ParameterList", "FunctionDef", "ReturnType", 
                "OrExpression", "CastExpression", "CastTarget", "SizeofExpr", "Sizeof", "ConditionalExpression", "GotoStatement", "SizeofOperand", 
                "InclusiveOrExpression", "WhileStatement", "SwitchStatement", "ContinueStatement", "Expression", "InitializerList"]

    ast_root = './data/graph/' + args.p + '/ast'
    ddg_cdg_root = './data/graph/' + args.p + '/ddg_cdg'
    max_tokens = []
    ddg_cdg_path_list = []
    max_edge_types = []
    node_tag = {}  # 标记存在边的顶点，剔除不存在边的点
    e_fp = open('./' + args.p + '_edges_list.txt', 'w', encoding='utf-8')
    ddg_cdg_path_list = get_svg_path(ddg_cdg_root, ddg_cdg_path_list)

    print('[+] extracting nodes and edges from svg files...')
    bar = tqdm.tqdm(ddg_cdg_path_list)
    for file_path in bar:
        '''生成ast、ddg和cdg的语料图'''
        dc_domTree = parse(file_path)
        dc_rootNode = dc_domTree.documentElement

        dc_node_list = dc_rootNode.getElementsByTagName('g')
        ast_name = file_path.split('/')[-1].split('.')[0] + '_ast.svg'
        ast_path = os.path.join(ast_root, ast_name)
        ast_domTree = parse(ast_path)
        ast_rootNode = ast_domTree.documentElement
        ast_node_list = ast_rootNode.getElementsByTagName('g')
        num_to_func = {}
        dc_tag = {}  
        ast_tag = {}  # 辅助ast补全ddg_cdg
        '''生成svg文件的node编号和code一一对应的dict文件'''
        for node in dc_node_list:
            if node.getAttribute('id')[:4] == 'node':
                num = node.getElementsByTagName('title')[0].childNodes[0].data
                t_list = node.getElementsByTagName('text')
                func_code = ''
                ntype = ''
                for t in t_list:
                    attr = t.childNodes[0].data
                    if attr[:4] == 'code':
                        func_code = attr[5:]
                        # max_tokens.append(func_code)
                        # if num not in num_to_func:
                        #     num_to_func[num] = func_code
                    if attr[:4] == 'type':
                        ntype = attr[5:]
                if ntype in key_ntypes0:  # 选定节点类型
                    max_tokens.append(func_code)
                    if num not in num_to_func:
                        num_to_func[num] = func_code

        for node in dc_node_list:  # ddg_cdg中边结构为s->d
            if node.getAttribute('id')[:4] == 'edge':
                s_d = node.getElementsByTagName('title')[0].childNodes[0].data
                source = s_d.split('->')[0]
                destination = s_d.split('->')[-1]
                if source in num_to_func and destination in num_to_func:
                    node_tag[num_to_func[source]] = 1
                    node_tag[num_to_func[destination]] = 1
                    dc_tag[source] = 1
                    dc_tag[destination] = 1
                    if node.getElementsByTagName('text'):
                        et = node.getElementsByTagName('text')[0].childNodes[0].data
                        max_edge_types.append(et)
        
        for node in ast_node_list:    # 标记跟cdg_ddg中有直接边的点
            if node.getAttribute('id')[:4] == 'edge':
                s_d = node.getElementsByTagName('title')[0].childNodes[0].data
                source = s_d.split('--')[0]
                destination = s_d.split('--')[-1]
                if source in dc_tag and destination not in dc_tag:
                    ast_tag[destination] = 1
                elif source not in dc_tag and destination in dc_tag:
                    ast_tag[source] = 1

        for node in ast_node_list:
            if node.getAttribute('id')[:4] == 'node':
                num = node.getElementsByTagName('title')[0].childNodes[0].data
                if num in ast_tag:
                    t_list = node.getElementsByTagName('text')
                    func_code = ''
                    ntype = ''
                    for t in t_list:
                        attr = t.childNodes[0].data
                        if attr[:4] == 'code':
                            func_code = attr[5:]
                            # max_tokens.append(func_code)
                            # if num not in num_to_func:
                            #     num_to_func[num] = func_code
                        if attr[:4] == 'type':
                            ntype = attr[5:]
                    if ntype in key_ntypes1:
                        max_tokens.append(func_code)
                        if num not in num_to_func:
                            num_to_func[num] = func_code

        for node in ast_node_list:
            if node.getAttribute('id')[:4] == 'edge':
                s_d = node.getElementsByTagName('title')[0].childNodes[0].data
                source = s_d.split('--')[0]
                destination = s_d.split('--')[-1]
                if source in dc_tag or destination in dc_tag:
                    if source in num_to_func and destination in num_to_func:
                        node_tag[num_to_func[source]] = 1
                        node_tag[num_to_func[destination]] = 1
        bar.update()
    bar.close()

    tokens = []
    for t in max_tokens:
        if node_tag.get(t) != None:
            tokens.append(t)
    tokens = dict(Counter(tokens))
    tokens = dict(sorted(tokens.items(), key=operator.itemgetter(1), reverse=True))
    max_edge_types = dict(Counter(max_edge_types))
    max_edge_types = dict(sorted(max_edge_types.items(), key=operator.itemgetter(1), reverse=True))
    nodes_vocab = {}
    edges_vocab = {}
    for i, item in enumerate(tokens.keys()):
        nodes_vocab[item] = i

    edges_vocab.update({'': 0})
    for i, item in enumerate(max_edge_types.keys()):
        edges_vocab[item] = i + 1
    
    print(len(nodes_vocab))
    with open(args.p + '_nodes_vocab.json', 'w') as fp:
        json.dump(nodes_vocab, fp)
    
    with open(args.p + '_edges_vocab.json', 'w') as fp:
        json.dump(edges_vocab, fp)

    '''生成函数的标记'''
    func_to_label = {}
    with open(args.p + '_labels.txt', 'r') as fp:
        label_lines = fp.readlines()
        for line in label_lines:
            func = line.split('@@')[0]
            label = int(line.split('@@')[-1])
            func_to_label[func.split('.')[0]] = label

    idx_to_node = {}
    features = []
    graph = []
    label = []
    '''生成图中边列表的txt文件'''
    print('[+] get edges list...')
    pbar = tqdm.tqdm(ddg_cdg_path_list)
    for file_path in pbar:
        dc_domTree = parse(file_path)
        dc_rootNode = dc_domTree.documentElement

        dc_node_list = dc_rootNode.getElementsByTagName('g')
        ast_name = file_path.split('/')[-1].split('.')[0] + '_ast.svg'
        ast_path = os.path.join(ast_root, ast_name)
        ast_domTree = parse(ast_path)
        ast_rootNode = ast_domTree.documentElement
        ast_node_list = ast_rootNode.getElementsByTagName('g')
        # 000cacf6f9dce7d71f88aadf7e9b3688eaa3ab69@qemu_12213@gen_sse.svg
        temp = file_path.split('/')[-1]
        temp = temp.split('@')
        file_name = temp[0] + '@' + temp[1]
        # node_list = []
        # node_list.extend(dc_node_list)
        # node_list.extend(ast_node_list)
        edge_msg = []  # 边信息[s, code, d]
        n_to_v = {}  # 将node编号转化为语料库中的编号
        node_tokens = []  # node中的code
        dc_tag = {}  # 标记ddg_cdg中与ast中有直接边的node(出现了结点编号不同，code相同的情况)
        ast_tag = {}  # 标记ast中与ddg_cdg中有直接边的node
        for node in dc_node_list:
            if node.getAttribute('id')[:4] == 'node':
                num = node.getElementsByTagName('title')[0].childNodes[0].data
                t_list = node.getElementsByTagName('text')
                func_code = ''
                ntype = ''
                for t in t_list:
                    attr = t.childNodes[0].data
                    if attr[:4] == 'code':
                        func_code = attr[5:]
                    if attr[:4] == 'type':
                        ntype = attr[5:]
                if ntype in key_ntypes0 and nodes_vocab.get(func_code) != None:  # 删去孤立的结点
                    node_tokens.append(func_code)
                    n_to_v[num] = str(nodes_vocab[func_code])
        
        edges_list = []
        for node in dc_node_list:    
            if node.getAttribute('id')[:4] == 'edge':
                s_d = node.getElementsByTagName('title')[0].childNodes[0].data
                source = s_d.split('->')[0]
                destination = s_d.split('->')[-1]
                if source in n_to_v and destination in n_to_v:  # 边节点都属于选定节点类型，保留边信息
                    edges_list.append(s_d)
                    dc_tag[source] = 1
                    dc_tag[destination] = 1
                    if node.getElementsByTagName('text'):
                        etype = node.getElementsByTagName('text')[0].childNodes[0].data
                        edge_msg.append([n_to_v[source], edges_vocab[etype], n_to_v[destination]])
                    else:
                        edge_msg.append([n_to_v[source], edges_vocab[''], n_to_v[destination]])
        
        for e in edges_list:
            node_num = e.split('->')
            e_fp.write(n_to_v[node_num[0]] + ' ' + n_to_v[node_num[-1]] + '\n')
        
        for node in ast_node_list:    # 标记跟cdg_ddg中有直接边的点
            if node.getAttribute('id')[:4] == 'edge':
                s_d = node.getElementsByTagName('title')[0].childNodes[0].data
                source = s_d.split('--')[0]
                destination = s_d.split('--')[-1]
                if source in dc_tag and destination not in dc_tag:
                    ast_tag[destination] = 1
                elif source not in dc_tag and destination in dc_tag:
                    ast_tag[source] = 1

        for node in ast_node_list:
            if node.getAttribute('id')[:4] == 'node':
                num = node.getElementsByTagName('title')[0].childNodes[0].data
                if num in ast_tag:
                    t_list = node.getElementsByTagName('text')
                    func_code = ''
                    ntype = ''
                    for t in t_list:
                        attr = t.childNodes[0].data
                        if attr[:4] == 'code':
                            func_code = attr[5:]
                        if attr[:4] == 'type':
                            ntype = attr[5:]
                    if ntype in key_ntypes1:
                        node_tokens.append(func_code)
                        n_to_v[num] = str(nodes_vocab[func_code])

        edges_list = []  # ast中边的列表
        for node in ast_node_list:    
            if node.getAttribute('id')[:4] == 'edge':
                s_d = node.getElementsByTagName('title')[0].childNodes[0].data
                source = s_d.split('--')[0]
                destination = s_d.split('--')[-1]
                if source in n_to_v and destination in n_to_v:
                    if (source in ast_tag and destination in dc_tag) or (source in dc_tag and destination in ast_tag):
                        edges_list.append(s_d)
                        edge_msg.append([n_to_v[source], edges_vocab[''], n_to_v[destination]])
                     
        for e in edges_list:
            node_num = e.split('--')
            e_fp.write(n_to_v[node_num[0]] + ' ' + n_to_v[node_num[-1]] + '\n')
        
        idx_to_node[file_name] = node_tokens
        features.append(node_tokens)
        graph.append(edge_msg)
        label.append(func_to_label[file_name])
        pbar.update()
    pbar.close()
    e_fp.close()
    
    # '''保存图嵌入特征'''
    # with open(args.p + '_graph_node_features.json', 'w') as fp:
    #     json.dump(idx_to_node, fp)

    # graph_pkl = {'features': features, 'graph': graph, 'label': label}
    # df = pd.DataFrame(graph_pkl)
    # df.to_pickle('./data/' + args.p + '_graph_input.pkl')

    '''embedding并存入到pkl文件中'''
    print('[+] get node features embedding...')
    G = nx.read_edgelist('./' + args.p + '_edges_list.txt', create_using=nx.DiGraph(), nodetype=str, data=None)
    # print(G.nodes())

    dp_model = deepwalk(G, walk_length=10, num_walks=80, workers=1)
    dp_model.train(window_size=5, iter=3)
    embeddings = dp_model.get_embedding()
    pkl.dump(embeddings, open(args.p + '_embeddings.pkl', 'wb'))