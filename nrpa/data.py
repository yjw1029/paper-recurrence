import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from config import WORDS_SIZE, REVIEWS_SIZE
import nltk


nltk.download('punkt')

with open('../data/yelp13/wordlist.txt', 'r', errors='replace') as f:
    WORD_PADDING = len(f.readlines())

# @paramas:
# id_file exp ../data/yelp13/usrlist.txt
# return dict like {"Alice": 0} or {"Bread": "0"}
# @usage:
# generate id_num
def init_id_dict(id_file):
    f = open(id_file, 'r', errors='replace')
    lines = f.readlines()
    dict = {}
    for i, id in enumerate(lines):
        if id[-1] == '\n':
            id = id[:-1]
        dict[id] = [i]
    f.close()
    return dict


# @paramas:
# vocab_file exp ../data
# return dict like {"hello": 0}
# @useage:
# generate index of single word
def init_voc_dict(vocab_file):
    global WORD_PADDING
    f = open(vocab_file, 'r', errors='replace')
    lines = f.readlines()
    word_number = len(lines)
    dict = {}
    for i, vocab in enumerate(lines):
        dict[vocab[:-1]] = i   

    WORD_PADDING = word_number
    f.close()
    return dict

# @params:
# review_file exp ../data/yelp13/train.txt
# user_file exp ../data/yelp13/usrlist.txt
# vocab_dict like {"hello": 0}
# return dict like {"Alice": [0, 1, 2, 3]} 
# list_value is the index of record review
# @usage:
# return all review index of users(key type str , value type list)
def review_devide_by_user(review_file, user_file):
    dict = {}

    # set key as user_name for dict
    with open(user_file, 'r', errors='replace') as f_user:
        for user in f_user.readlines():
            if user[-1] == '\n':
                user = user[:-1]
            dict[user] = []
    
    with open(review_file, 'r', errors='replace') as f_review:
        reviews = f_review.readlines()
        for index in range(len(reviews)):
            info = reviews[index].split('\t\t')
            # print(info)
            user_name = info[0]
            dict[user_name].append(index)
    return dict


# @params:
# review_file exp ../data/yelp13/train.txt
# user_file exp ../data/yelp13/prdlist.txt
# vocab_dict like {"hello": 0}
# return dict like {"Bread": [[1,2,3], [0,2,3]]}
# @usage:
# return all word vectors of items(key type str , value type array3d)
def review_devide_by_item(review_file, item_file):
    dict = {}

     # set key as user_name for dict
    with open(item_file, 'r', errors='replace') as f_item:
        for item in f_item.readlines():
            if item[-1] == '\n':
                item = item[:-1]
            dict[item] = []
    
    with open(review_file, 'r', errors='replace') as f_review:
        reviews = f_review.readlines()
        for index in range(len(reviews)):
            info = reviews[index].split('\t\t')
            # print(info)
            item_name = info[1]
            dict[item_name].append(index)
    return dict


class NRPADataset(Dataset):
    def __init__(self, review_file, user_file, item_file, wordlist_file):
        super(NRPADataset, self).__init__()
        self.review_file = review_file
        self.user_file = user_file
        self.item_file = item_file
        self.wordlist_file = wordlist_file
        self.user_dict = init_id_dict(user_file)
        self.item_dict = init_id_dict(item_file)
        self.voc_dict = init_voc_dict(wordlist_file)
        self.review_user_dict = review_devide_by_user(review_file, user_file)
        self.review_item_dict = review_devide_by_item(review_file, item_file)

    def get_vec_of_review(self, review: str, padding = WORD_PADDING, vec_len=WORDS_SIZE):
        review_word_list =  nltk.word_tokenize(review)
        len_sent_list = len(review_word_list)
        review_vec = []

        for i in range(vec_len):
            if i < len_sent_list:
                try:
                    vocab_index = self.voc_dict[review_word_list[i]]
                except KeyError:
                    vocab_index = padding
            else:
                vocab_index = padding
            review_vec.append(vocab_index)
        return review_vec


    def get_review_tensor_from_list(self, name, isuser = True, review_vec_len = REVIEWS_SIZE):
        if isuser:
            review_dict = self.review_user_dict
        else:
            review_dict = self.review_item_dict        
        review_index_list = review_dict[name]

        with open(self.review_file, 'r', errors="replace") as f_review:
            records = f_review.readlines()
            len_review_index_list = len(review_index_list)
            review_tensor = []
            for i in range(review_vec_len):
                if i < len_review_index_list:
                    info = records[i].split('\t\t')
                    review = info[3]
                    if review[-1] == '\n':
                        review = review[:-1] 
                else: 
                    review = ''
                review_vec = self.get_vec_of_review(review)
                review_tensor.append(review_vec)
        return torch.LongTensor(review_tensor)
        

    def __getitem__(self, index):
        with open(self.review_file, 'r', errors="replace") as f_review:
            info = f_review.readlines()[index].split('\t\t')
            user_name = info[0]
            item_name = info[1]
            rate = int(info[2])
            user_id_vector =  torch.LongTensor(self.user_dict[user_name])
            item_id_vector = torch.LongTensor(self.item_dict[item_name])
            user_review_vectors = self.get_review_tensor_from_list(user_name)
            item_review_vectors = self.get_review_tensor_from_list(item_name, isuser=False)
            rate = torch.FloatTensor([rate])

        return (user_review_vectors, user_id_vector, item_review_vectors, item_id_vector, rate)


    def __len__(self):
        f = open(self.review_file, 'r', errors="replace")
        return len(f.readlines())


