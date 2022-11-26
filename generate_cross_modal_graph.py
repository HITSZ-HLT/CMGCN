import numpy as np
import spacy
import pickle
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from pytorch_pretrained import BertTokenizer
from nltk.corpus import wordnet as wn

tokenizer = BertTokenizer.from_pretrained("./bert_base_uncased")
nlp = spacy.load('en_core_web_sm')
y = 3
with open("./processed_data/vit_features.B32.finetuned.pkl",'rb') as fin:
    boxes = pickle.load(fin)

def dependency_adj_matrix_2(text):
    doc = nlp(text)
    mat = defaultdict(list,[])
    for t in doc:
        for child in t.children:
            mat[child.i].append(t.i)
            mat[t.i].append(child.i)
    return mat

def load_sentic_word():
    """
    load senticNet
    """
    path = './senticNet/senticnet_word.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = float(sentic)
    fp.close()
    return senticNet

senticNet = load_sentic_word()

def get_sentic_score(word_i,word_j):
    if word_i not in senticNet or word_j not in senticNet or word_i == word_j:
        return 0
    return abs(float(senticNet[word_i] - senticNet[word_j])) * y**(-1*senticNet[word_i]*senticNet[word_j])

def generate_graph(line):
    line = line.lower().strip()
    bert_token = tokenizer.tokenize(line)
    document = nlp(line)
    spacy_token = [str(x) for x in document]
    spacy_len = len(spacy_token)
    bert_len = len(bert_token)
    outter_graph = np.zeros((bert_len, bert_len)).astype('float32')
    split_link = [None]*spacy_len

    ii = 0
    jj = 0
    pre = []
    s = ""
    while ii<bert_len and jj < spacy_len:
        bert_ = bert_token[ii].replace("##","")
        spacy_ = spacy_token[jj]
        s += bert_
        pre.append(ii)
        if s == spacy_:
            split_link[jj] = deepcopy(pre)
            pre = []
            s = ""
            jj += 1        
        ii += 1
    flag = False
    mat = dependency_adj_matrix_2(line)
    if not(ii<bert_len or jj < spacy_len):
        for key,linked in mat.items():
                try:
                    for x in split_link[int(key)]:
                        for link in linked:
                            for y in split_link[int(link)]:
                                outter_graph[x][y] = 1     
                except:
                    flag = True
                    break
    else:
        flag = True
    if flag:
            tokens = spacy_token
            outter_graph = np.zeros((spacy_len, spacy_len)).astype('float32')
            inner_graph = np.identity(spacy_len).astype('float32')
            doc = nlp(line)
            for token in doc:
                for child in token.children:
                    outter_graph[token.i][child.i] = 1
                    outter_graph[child.i][token.i] = 1    
    else:
            tokens = bert_token
            inner_graph = np.identity(bert_len).astype('float32')
            for link in split_link:
                for x in link:
                    for y in link:
                        inner_graph[x][y] = 1

    
    outter_graph = np.pad(outter_graph,((1,1),(1,1)),'constant')
    inner_graph = np.pad(inner_graph,((1,1),(1,1)),'constant')
    inner_graph[0][0] = 1
    inner_graph[-1][-1] = 1
    graph1 = inner_graph + outter_graph
    for i in range(len(graph1)):
        for j in range(len(graph1)):
            if graph1[i][j] > 0:
                graph1[i][j] = 1
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    return graph1,tokens,flag

def get_split(line):
    line = line.lower().strip()
    bert_token = tokenizer.tokenize(line)
    document = nlp(line)
    spacy_token = [str(x) for x in document]
    spacy_len = len(spacy_token)
    bert_len = len(bert_token)
    outter_graph = np.zeros((bert_len, bert_len)).astype('float32')
    split_link = [None]*spacy_len
    ii = 0
    jj = 0
    pre = []
    s = ""
    while ii < bert_len and jj < spacy_len:
        bert_ = bert_token[ii].replace("##","")
        spacy_ = spacy_token[jj]
        s += bert_
        pre.append(ii)
        if s == spacy_:
            split_link[jj] = deepcopy(pre)
            pre = []
            s = ""
            jj += 1        
        ii += 1
    mat = dependency_adj_matrix_2(line)
    if not(ii<bert_len or jj < spacy_len):
        for key,linked in mat.items():
                try:
                    for x in split_link[int(key)]:
                        for link in linked:
                            for y in split_link[int(link)]:
                                outter_graph[x][y] = 1     
                except:
                    break
    return split_link

def generate_image_graph(text,box_texts,flag):
    box_tokens = [list(generate_graph(text))+[text] for text in box_texts]
    box_tokens = [x[1:] for x in box_tokens]
    for idx,val in enumerate(box_tokens):
        box_tokens[idx] = [val[0][1:-1]] + val[1:]
    box_texts_len = sum([len(x[0]) for x in box_tokens])
    bert_text_tokens = tokenizer.tokenize(text.lower())
    spacy_text_tokens = nlp(text.lower())
    spacy_text_tokens = [str(x).lower() for x in spacy_text_tokens]

    flags = []
    word = []
    if flag:
        graph = np.zeros((len(spacy_text_tokens), (box_texts_len))).astype('float32')
        for i,token_i in enumerate(spacy_text_tokens):
            cur = 0
            si = wn.synsets(token_i)
            if len(si) == 0:
                continue
            si = si[0]
            for tokens,flag,text in box_tokens:
                flags.append(flag)
                if flag:
                    for j,token_j in enumerate(tokens):
                        sj = wn.synsets(token_j)
                        if i == 0:
                            word.append([token_j])
                        if len(sj) == 0:
                            cur += 1
                            continue
                        sj = sj[0]
                        graph[i][cur] = wn.path_similarity(si,sj) + get_sentic_score(si,sj)
                        cur += 1
                        
                else:
                    split_link1 = get_split(text)
                    tokens_ = nlp(text)
                    tokens_ = [str(x) for x in tokens_]
                    
                    for j,token_j in enumerate(tokens_):
                        sj = wn.synsets(token_j)
                        if i == 0:
                            tmp = []
                        for t in split_link1[j]:
                            if i == 0:
                                tmp.append(tokens[j])
                            if len(sj) == 0:
                                cur += 1
                                continue
                            sj_ = sj[0]
                            graph[i][cur] = wn.path_similarity(si,sj_) + get_sentic_score(si,sj_)
                            cur += 1
                        if i == 0:
                            word.append(tmp)
                        if len(sj) == 0:
                            continue

    else:
        graph = np.zeros((len(bert_text_tokens), (box_texts_len))).astype('float32')
        split_links_i_ = get_split(text)
        split_links_i = dict()
        for idx,value in enumerate(split_links_i_):
            for v in value:
                split_links_i[bert_text_tokens[v]] = spacy_text_tokens[idx]
        for i,token in enumerate(bert_text_tokens):
            cur = 0
            token = split_links_i[token]
            si = wn.synsets(token)
            if len(si) == 0:
                continue
            si = si[0]
            
            for tokens,flag,text in box_tokens:
                    flags.append(flag)
                    if flag:
                        for j,token_j in enumerate(tokens):
                            sj = wn.synsets(token_j)
                            if i == 0:
                                word.append([token_j])
                            if len(sj) == 0:
                                cur += 1
                                continue
                            sj = sj[0]
                            graph[i][cur] = wn.path_similarity(si,sj) + get_sentic_score(si,sj)
                            cur += 1            
                    else:
                        split_link1 = get_split(text)
                        tokens_ = nlp(text)
                        tokens_ = [str(x) for x in tokens_]
                        if i == 0:
                            tmp = []
                        for j,token_j in enumerate(tokens_):
                            sj = wn.synsets(token_j)
                            for t in split_link1[j]:
                                if i == 0:
                                    tmp.append(tokens[j])
                                if len(sj) == 0:
                                    cur += 1
                                    continue
                                sj_ = sj[0]
                                graph[i][cur] = wn.path_similarity(si,sj_) + get_sentic_score(si,sj_)
                                cur += 1
                                
                            if i == 0:
                                word.append(tmp)
                            if len(sj) == 0:
                                continue
    return np.pad(graph,((1,1),(1,1)),'constant'),flags,word

N = 10

def process(filename,outfile = ""):
    with open("./data/{}.txt".format(filename),'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        cnt = 0
        lines = [x.strip() for x in lines]
        dic = dict()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            data = eval(line)
            if 'train' in filename:
                id_,text,label = data
            else:
                id_,text,label1,label = data
            if id_ in boxes:
                box_text = list(boxes[id_].items())
                box_text = [(" ".join(key.split("_")[:-1]),int(key.split("_")[-1]),val) for key,val in box_text]
                box_text = sorted(box_text,key = lambda x:x[1])
                box_text = [(x,z) for x,y,z in box_text]
                
                box_text = box_text[:N]
                box_vit = [x[1][0] for x in box_text]
                box_token = [tokenizer.tokenize(text.lower()) for text,_ in box_text]

                text_graph,tokens,flag = generate_graph(text.lower())
                image_graph,flags,word = generate_image_graph(text.lower(),[x[0] for x in box_text],flag)
                

                dic[id_] = {'text':text,'label':int(label),'tokens':tokens,'graph':text_graph,\
                        'box_tokens':box_token,'box_vit':box_vit,'image_graph':image_graph,"flags":flags,\
                            "flag":flag,'word':word,"box_text":box_text}
                
        print("{} : {} {}".format(filename,len(dic),cnt))
        pickle.dump(dic,open("{}{}.data".format(filename,outfile),'wb'))

process("train.cleaned",".only_box_top{}_new_2".format(N))
process("test.cleaned",".only_box_top{}_new_2".format(N))
process("valid.cleaned",".only_box_top{}_new_2".format(N))