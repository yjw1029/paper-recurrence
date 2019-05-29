import bisect
import nltk
# from data import init_voc_dict, review_devide_by_user, review_devide_by_item

# word_level dim predict
def count_word_size():
    with open("../data/yelp13/train.txt", 'r', errors="replace") as  f_train:
        word_len = []

        for line in f_train.readlines():
            info = line[:-1].split('\t\t') 
            try:  
                sent_list = nltk.sent_tokenize(info[3])
            except IndexError:
                sent_list = []
                print(info)
            for sent in sent_list:
                word_list = nltk.word_tokenize(sent)
                bisect.insort(word_len, len(word_list))

        allnum = len(word_len)
        # print(word_len)
        print("-----------word size-------------")
        print("100%:", word_len[allnum -1])
        print("98%:", word_len[int(allnum * 0.98)])
        print("98%:", word_len[int(allnum * 0.97)])
        print("95%:", word_len[int(allnum * 0.95)])
        print("93%:", word_len[int(allnum * 0.93)])
        print("90%:", word_len[int(allnum * 0.90)])
        print("88%:", word_len[int(allnum * 0.88)])
        print("85%:", word_len[int(allnum * 0.85)])


# sentence_level predict
def count_sent_size():
    with open("../data/yelp13/train.txt", 'r', errors="replace") as  f_train:
        sent_len = []

        for line in f_train.readlines():
            info = line[:-1].split('\t\t') 
            try:  
                sent_list = nltk.sent_tokenize(info[3])
            except IndexError:
                sent_list = []
                print(info)
            bisect.insort(sent_len, len(sent_list))

        allnum = len(sent_len)
        print("-----------sent size-------------")
        print("100%:", sent_len[allnum -1])
        print("98%:", sent_len[int(allnum * 0.98)])
        print("98%:", sent_len[int(allnum * 0.97)])
        print("95%:", sent_len[int(allnum * 0.95)])
        print("93%:", sent_len[int(allnum * 0.93)])
        print("90%:", sent_len[int(allnum * 0.90)])
        print("88%:", sent_len[int(allnum * 0.88)])
        print("85%:", sent_len[int(allnum * 0.85)])


# review_level dim predict

def count_review_num(dict):
    len_list = []
    for i in dict:
        review_num = len(dict[i])
        bisect.insort(len_list, review_num)

    num = len(len_list)
    print("allnum:", num)
    print("95%:", len_list[num // 20 * 19])
    print("90%:", len_list[num // 10 * 9])
    print("85%:", len_list[num // 20 * 17])


if __name__ == "__main__":
    print("-------------- sent --------------")
    count_sent_size()
    print("-------------- word --------------")
    count_word_size()
    # dir = "../data/yelp13/"
    # voc_dict = init_voc_dict(dir+"wordlist.txt")
    # dict_user = review_devide_by_user(dir+"train.txt", dir+"usrlist.txt",voc_dict)
    # print("-------------- user --------------")
    # count_review_num(dict_user)
    # dict_item = review_devide_by_item(dir+"train.txt", dir+"prdlist.txt",voc_dict)
    # print("-------------- item --------------")
    # count_review_num(dict_item)
