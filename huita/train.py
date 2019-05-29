import torch
from huita import HUITA
from data import HUITADataset
import torch.utils.data as Data 
import os

EPOCH = 10
BATCH_SIZE = 10
NET_WORKERS = 5
learning_rate = 0.001

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dir = "../data/yelp13/"
train_set = HUITADataset(dir+"train.txt", dir+"usrlist.txt", dir+"prdlist.txt", dir+"wordlist.txt")
test_set = HUITADataset(dir+"test.txt", dir+"usrlist.txt", dir+"prdlist.txt", dir+"wordlist.txt")

def collate_fn(data):
    # user_review_vectors, user_id_vector, item_review_vectors, item_id_vector
    data = list(zip(*data))
    
    rslt_user_review = torch.stack(data[0], dim=0)
    rslt_user_id = torch.stack(data[1], dim=0)
    rslt_item_review = torch.stack(data[2], dim=0)
    rslt_item_id = torch.stack(data[3], dim=0)
    rslt_rate = torch.stack(data[4], dim=0)

    return rslt_user_review, rslt_user_id, rslt_item_review, rslt_item_id, rslt_rate


train_loader = Data.DataLoader(train_set, batch_size = BATCH_SIZE, collate_fn = collate_fn, shuffle=True, num_workers=NET_WORKERS)
test_loader = Data.DataLoader(test_set, batch_size = BATCH_SIZE, collate_fn = collate_fn, shuffle=True, num_workers=NET_WORKERS)

print("----------------------- finish data load -----------------------")

# init sequece: weights_matrix_user, weights_matrix_item, uid_matrix_len, iid_matrix_len
huita_para = torch.load('embed_weight.pkl')
net = HUITA(huita_para['weights_matrix_user'],
           huita_para['weights_matrix_item'],
           huita_para['uid_matrix_len'],
           huita_para['iid_matrix_len']).to(device)

print("----------------------- finish net load -----------------------")

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
loss_func = torch.nn.MSELoss()

# train net function
def train_net(loader):
    running_loss = 0
    for epoch in range(EPOCH):
        scheduler.step()
        net.train(True)
        for step, (user_review_vec, user_id_vec, item_review_vec, item_id_vec, rate) in enumerate(loader):
            user_review_vec= user_review_vec.to(device)
            user_id_vec = user_id_vec.to(device)
            item_review_vec = item_review_vec.to(device)
            item_id_vec = item_id_vec.to(device)
            rate = rate.to(device)
            optimizer.zero_grad()
            # param: all_user_reviews, all_item_reviews, user_id_num, item_id_num
            predicted_rate = net(user_review_vec, item_review_vec, user_id_vec, item_id_vec)
            loss = loss_func(predicted_rate, rate)
            loss.backward()
            optimizer.step() 
                
            running_loss += loss.item()
            if step % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 200))
                running_loss = 0.0
        if os.path.exists('Model/HUITA{}.pkl'.format(epoch)):
            os.remove('Model/HUITA{}.pkl'.format(epoch))
        torch.save(net, 'Model/HUITA{}.pkl'.format(epoch))

print("------------- start train -------------")
train_net(train_loader)

# start measure accuracy
net.eval()
running_loss = 0
cnt = 0
with torch.no_grad():
    for step, (user_review_vec, user_id_vec, item_review_vec, item_id_vec, rate) in enumerate(test_loader):
        user_review_vec= user_review_vec.to(device)
        user_id_vec = user_id_vec.to(device)
        item_review_vec = item_review_vec.to(device)
        item_id_vec = item_id_vec.to(device)
        rate = rate.to(device)
        predicted_rate = net(user_review_vec, user_id_vec, item_review_vec, item_id_vec)
        loss = loss_func(predicted_rate, rate)
        running_loss += loss.item()
        cnt += 1

running_loss = running_loss / cnt
        
print("test ave MSE loss: {}".format(running_loss))

print('measure finished')

# use last test_loader to train
train_net(test_loader)
if os.path.exists('Model/HUITA_last.pkl'):
    os.remove('Model/HUITA_last.pkl')
torch.save(net, 'Model/HUITA_last.pkl')

print('all finished')
