import torch
# from nrpa import NRPA
from nrpa.data import NRPADataset
import torch.utils.data as Data 

EPOCH = 10
BATCH_SIZE = 8
CLASS_NUM = 2
NET_WORKERS = 3
learning_rate = 0.001

dir = "../data_process/fake_data/"
data_set = NRPADataset(dir+"train.txt", dir+"usrlist.txt", dir+"prdlist.txt", dir+"wordlist.txt")

def collate_fn(data):
    # user_review_vectors, user_id_vector, item_review_vectors, item_id_vector
    data = list(zip(*data))
    
    rslt_user_review = torch.stack(data[0], dim=0)
    rslt_user_id = torch.stack(data[1], dim=0)
    rslt_item_review = torch.stack(data[2], dim=0)
    rslt_item_id = torch.stack(data[3], dim=0)
    rslt_rate = torch.stack(data[4], dim=0)

    return rslt_user_review, rslt_user_id, rslt_item_review, rslt_item_id, rslt_rate


data_loader = Data.DataLoader(data_set, batch_size = 2, collate_fn = collate_fn)

for i in enumerate(data_loader):
    print("----------------- start ------------------")
    print(i)

# S = NRPA()
