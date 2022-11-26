import spacy
import pickle
from tqdm import tqdm
import numpy as np
sp_nlp = spacy.load('en_core_web_sm')

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
        for i in tqdm(range(len(text))):
            x = text[i]
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

glove_tokenizer = Tokenizer()
corpus = []
for filename in ['train.txt','test2.txt','valid2.txt']:
    filename = "./data/" + filename
    with open(filename,'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            list_ = eval(line)
            corpus.append(list_[1].lower().strip())
glove_tokenizer.fit_on_text(corpus)
with open('glove_word2idx.pkl', 'wb') as f:
    pickle.dump(glove_tokenizer.word2idx, f)

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
                embedding_matrix[i] = vec
        return embedding_matrix

embedding_matrix = build_embedding_matrix(glove_tokenizer.word2idx,300)

with open('embedding_matrix.pkl', 'wb') as f:
    pickle.dump(embedding_matrix, f)