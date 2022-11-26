import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from pytorch_pretrained import BertTokenizer
import spacy

sp_nlp = spacy.load('en_core_web_sm')


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_raw = lines[i].lower().strip()
                try:
                    entity, attribute = lines[i + 1].lower().strip().split()
                except:
                    entity = lines[i + 1].lower().strip()
                    attribute = ''
                text += text_raw + entity + attribute + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname = None):
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['UNK'] = self.idx
            self.idx2word[self.idx] = 'UNK'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):   
        words = []
        for x in text:
            x = x.lower().strip()
            xx = sp_nlp(x)
            words = words + [str(y) for y in xx]
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower().strip()
        words = sp_nlp(text)
        words = [str(x) for x in words]

        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
    
    def text_to_sequence_(self,indexs, reverse=False, padding='post', truncating='post',pad = False,truncate = 512):
        sequence = self.tokenizer.convert_tokens_to_ids(indexs)
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return sequence


class Dataset(Dataset):
    def __init__(self, fname, tokenizer):
        class Dataset_(object):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)

        import pickle
        data = pickle.load(open(fname,'rb'))

        print("{}.data".format(fname))
        all_data = []
        for key,value in data.items():
            label = int(value['label'])
            graph = value['graph'] + value["sentic_graph"]
            tokens = value['tokens']
            box_tokens = value['box_tokens']
            box_vit = value["box_vit"]
            image_graph = value["image_graph"]

            bert_indices = tokenizer.text_to_sequence_(tokens)
            box_indices = [tokenizer.text_to_sequence_(token) for token in box_tokens]

            data_ = {
                'label':label,
                'graph':graph,
                'bert_indices':bert_indices,
                'box_indices':box_indices,
                'box_vit':box_vit,
                'image_graph':image_graph,
            }
            all_data.append(data_)
        self.data = Dataset_(all_data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
