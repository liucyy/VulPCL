import sys
import os
from collections import Counter
import tqdm
import argparse
import json
import pickle as pkl
import operator
import pandas as pd
import numpy as np
from xml.dom.minidom import parse
sys.path.append('.')

MAX__VOCAB_LEN = 10000
SPECIAL_WORD = {'<PAD>':0, '<UNK>':1, '<CLS>':2, '<SEP>':3, '<MASK>':4}

def get_svg_path(root_path, file_path_list):
    path_list = os.listdir(root_path)
    for path in path_list:
        co_path = os.path.join(root_path, path)
        if os.path.isfile(co_path) and co_path.endswith('.svg'):
            file_path_list.append(co_path)
        elif os.path.isdir(co_path):
            get_svg_path(co_path, file_path_list)
    return file_path_list


if __name__ == '__main__':
    # file = open('./data/data_svg/01/00a1e1337f22376909338a5319a378b2e2afdde8@FFmpeg_6961@mmap_read_frame.svg', 'r')
    # fp = file.read()
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str, default='FFmpeg', help='input project name')
    args = parser.parse_args()

    '''保存漏洞行信息'''
    func_to_line_index = {}
    with open('big_vul_msg.txt', 'r') as fp:
        big_vul_msg = fp.read().split('\n--------------------------\n')
        for line in big_vul_msg:
            if line == '':
                continue
            # print(line.split('&&'))
            func_id = line.split('&&')[0][:-2]
            vul_line_index = line.split('&&')[3]
            if vul_line_index != 'nan':
                vul_line_index = vul_line_index.split(',')
                func_to_line_index[func_id] = vul_line_index
    print(len(func_to_line_index))

    '''文件名对应标签'''
    fp = open(args.p + '_labels.txt', 'r').read().split('\n')
    file_label = {}
    for f in fp:
        f = f.split('@@')
        file_label[f[0][:-2]] = f[-1]
        if f[0][:-2] not in func_to_line_index:
            func_to_line_index[f[0][:-2]] = []

    root_path = './data/graph/' + args.p + '/cfg_dfg'

    keys_1 = ['\t', '\n', ' ', '!', '!=', '"', '#', '#define', '#elif', '#else', '#endif', '#if', '#ifdef', '#ifndef', 
            '#include', '%', '%=', '&', '&&', '&=', "'", '(', ')', '*', '*=', '+', '++', '+=', ',', '-', '--', '-=', 
            '->', '.', '...', '/', '/=', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '::', ';', '<', '<<', 
            '<<=', '<=', '=', '==', '>', '>=', '>>', '>>=', '?', 'A', 'B', 'C', 'D', 'E', 'ERROR', 'F', 'G', 'H', 'I', 
            'J', 'K', 'L', 'L"', "L'", 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', 
            ']', '^', '^=', '_', '__attribute__', 'a', 'b', 'break', 'c', 'call_expression', 'case', 'class', 'comment', 
            'const', 'continue', 'd', 'default', 'defined', 'delete', 'do', 'e', 'else', 'enum', 'extern', 'f', 'false', 
            'for', 'g', 'goto', 'h', 'i', 'if', 'inline', 'j', 'k', 'l', 'long', 'm', 'n', 'new', 'null', 'o', 'p', 'q', 
            'r', 'register', 'restrict', 'return', 's', 'short', 'signed', 'sizeof', 'static', 'struct', 'switch', 't', 
            'this', 'true', 'try', 'typedef', 'u', 'union', 'unsigned', 'v', 'volatile', 'w', 'while', 'x', 'y', 'z', '{',
            '|', '|=', '||', '}', '~']

    keys_2 = ["alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit",
            "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
            "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const",
            "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
            "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast",
            "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
            "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
            "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "reflexpr",
            "register", "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static",
            "static_assert", "static_cast", "struct", "switch", "synchronized", "template", "this",
            "thread_local", "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned",
            "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq", "NULL"]
    
    key_ntypes = ["ExpressionStatement", "Condition", "IdentifierDeclStatement", "Parameter", "BreakStatement", "ReturnStatement", 
                "GotoStatement", "ForInit", "IncDecOp", "ContinueStatement", "AssignmentExpr",  "ClassDefStatement", "Expression"]

    tokens = []
    file_path_list = []
    file_path_list = get_svg_path(root_path, file_path_list)
    # print(file_path_list)
    id = []
    vul_index = []
    functions = []
    feature1 = []
    feature2 = []
    label = []
    g_data = json.load(open(args.p + '_graph_node_features.json'))
    print('[+] extracting token features from svg file...')
    bar = tqdm.tqdm(file_path_list)

    '''获取源代码id及其对应代码'''
    id_to_funcs = {}
    source_code_path = './data/code/' + args.p
    code_path = os.listdir(source_code_path)
    for cp in code_path:
        func_name = cp.split('.')[0]
        s_code = ''
        # print(cp)
        with open('./data/code/' + args.p + '/' + cp, 'r') as fp:
            s_code = fp.read()
        id_to_funcs[func_name] = s_code

    for file_path in bar:
        '''生成顶点code的语料库'''
        file_name = file_path.split('/')[-1]
        file_name = file_name.split('@')
        f_id = file_name[0] + '@' + file_name[1]
        if f_id not in g_data:
            continue
        id.append(f_id)
        label.append(file_label[f_id])

        domTree = parse(file_path)
        rootNode = domTree.documentElement
        # print(rootNode.nodeName)

        node_list = rootNode.getElementsByTagName("g")
        f_tokens = []
        '''生成svg文件中的node编号和code一一对应的dict文件'''
        for node in node_list:
            if node.getAttribute("id")[:4] == 'node':
                num = node.getElementsByTagName("title")[0].childNodes[0].data
                t_list = node.getElementsByTagName('text')
                func_code = ''
                ntype = ''
                for line in t_list:
                    # print(line.nodeName, " ", type(line.childNodes[0].data), line.childNodes[0].data)
                    attr = line.childNodes[0].data
                    if attr[:4] == 'code':
                        func_code = attr[5:].split(' ')
                    if attr[:4] == 'type':
                        ntype = attr[5:]
                if ntype in key_ntypes:
                    f_tokens.extend(func_code)
        
        tokens.extend(f_tokens)
        vul_index.append(func_to_line_index[f_id])
        functions.append(id_to_funcs[f_id])
        feature1.append(f_tokens)
        feature2.append(g_data[f_id])
        bar.update()
    bar.close()
    
    tokens.extend(keys_1)
    tokens.extend(keys_2)

    tokens = dict(Counter(tokens))
    tokens = dict(sorted(tokens.items(), key=operator.itemgetter(1), reverse=True))
    nodes_vocab = dict()
    nodes_vocab.update(SPECIAL_WORD)

    for i, item in enumerate(tokens.keys()):
        nodes_vocab[item] = i + 5

    with open(args.p + '_code_vocab.json', 'w') as fp:
        json.dump(nodes_vocab, fp)

    print(len(nodes_vocab))

    '''保存词嵌入特征'''
    # with open(args.p + '_word_node_features.json', 'w') as fp:
    #     json.dump(idx_to_node, fp)

    '''生成code整体的pkl文件'''
    print('[+] generate graph_input.pkl...')
    graph_pkl = {'id':id, 'code': functions, 'feature1': feature1, 'feature2': feature2, 'label': label, 'vul_index': vul_index}
    df = pd.DataFrame(graph_pkl)
    df.to_pickle('./data/' + args.p + '_graph_input.pkl')
    