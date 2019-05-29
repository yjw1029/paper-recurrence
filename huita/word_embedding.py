from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec
from config import WORD_EMBEDDING_DIM, ID_EMBEDDING_DIM
import numpy as np
import os
import torch


dir_dataset = "../data/yelp13/"

gensim_model = KeyedVectors.load_word2vec_format('../data_process/GoogleNews-vectors-negative300.bin', binary=True)

f_vocab = open(dir_dataset + 'wordlist.txt', 'r', errors="replace")
target_vocab = []
words_found = 0

# prepare vocabulary list
for line in f_vocab.readlines():
    target_vocab.append(line[:-1])

review_matrix_len = len(target_vocab)

# add 1 for padding
weights_matrix_user = np.zeros((review_matrix_len+1, WORD_EMBEDDING_DIM))
weights_matrix_item = np.zeros((review_matrix_len+1, WORD_EMBEDDING_DIM))


# prepare the weights_matrix of embedding in user and item net
for i, word in enumerate(target_vocab):
    try: 
        weights_matrix_user[i] = gensim_model[word]
        weights_matrix_item[i] = gensim_model[word]
        words_found += 1
    except KeyError:
        weights_matrix_user[i] = np.random.normal(scale=0.6, size=(WORD_EMBEDDING_DIM, ))
        weights_matrix_item[i] = np.random.normal(scale=0.6, size=(WORD_EMBEDDING_DIM, ))

# generate for padding
weights_matrix_item[review_matrix_len] = np.random.normal(scale=0.6, size=(WORD_EMBEDDING_DIM, ))
weights_matrix_user[review_matrix_len] = np.random.normal(scale=0.6, size=(WORD_EMBEDDING_DIM, ))

# np array to tensor
weights_matrix_user = torch.FloatTensor(weights_matrix_user)
weights_matrix_item = torch.FloatTensor(weights_matrix_item)

f_userid = open(dir_dataset + "usrlist.txt", 'r', errors="replace")
f_itemid = open(dir_dataset + "prdlist.txt", 'r', errors="replace")
uid_matrix_len = len(f_userid.readlines())
iid_matrix_len = len(f_itemid.readlines())



if __name__ == "__main__":
    if os.path.exists('./embed_weight.pkl'):
        os.remove('./embed_weight.pkl')
    print(weights_matrix_user)
    print("-----------------------------")
    print(weights_matrix_item)
    torch.save({'weights_matrix_user': weights_matrix_user,
                'weights_matrix_item': weights_matrix_item,
                'uid_matrix_len':uid_matrix_len,
                'iid_matrix_len':iid_matrix_len}, './embed_weight.pkl')

