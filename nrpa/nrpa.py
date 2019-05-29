import numpy as np
import torch
from torch import nn
import torch.nn.functional as F 
import config
# from word_embedding import weights_matrix_user, weights_matrix_item, uid_matrix_len, iid_matrix_len


class reviewEncoder(nn.Module):
    def __init__(self, word_weight_matrix, id_matrix_len):
    # Args: 
    # word_weight_matrix: pretrained review word embedding
    # id_matrix_len: num of ids
        super(reviewEncoder, self).__init__()
        word_num_embeddings, word_embedding_dim = word_weight_matrix.size()
        self.review_embeds = nn.Embedding(word_num_embeddings, word_embedding_dim)
        self.review_embeds.load_state_dict({'weight': word_weight_matrix})

        self.id_embeds = nn.Embedding(id_matrix_len, config.ID_EMBEDDING_DIM)
    
        self.conv = nn.Conv1d(
            in_channels = config.WORD_EMBEDDING_DIM,
            out_channels = config.NUM_OF_FILTERS, 
            kernel_size = config.WINDOW_SIZE_CNN,
            padding = (config.WINDOW_SIZE_CNN -1) //2
            )
        self.l1 = nn.Linear(config.ID_EMBEDDING_DIM, config.ATTEN_VEC_DIM)
        self.A1 = nn.Parameter(torch.randn(config.ATTEN_VEC_DIM, config.NUM_OF_FILTERS), requires_grad=True)

    # paramas:  (LongTensor, LongTensor)
    # sentence_vector: (batch * review_size) * word_size
    # id_vector: batch * 1 
    def forward(self, sentence_vector, id_num):

        # embeds id and sentence
        # sentence_matrix: batch * word_size * word_embedding_size
        # id_vector:batch * 1 * id_embedding_size
        sentence_vector = sentence_vector.view(-1)
        sentence_matrix = self.review_embeds(sentence_vector).view(-1, config.WORDS_SIZE, config.WORD_EMBEDDING_DIM)
        sentence_matrix = sentence_matrix.permute(0, 2, 1)

        id_vector = self.id_embeds(id_num.view(-1)).view(-1, 1, config.ID_EMBEDDING_DIM)

        # conv get C(batch * filter_size * word_size)
        C = F.relu(self.conv(sentence_matrix))

        # MLP get qw(batch * 1 * atte_v_szie)
        qw = F.relu(self.l1(id_vector))

        # get attention weight
        qw_temp = qw.contiguous().view(-1, config.ATTEN_VEC_DIM)
        
        g = torch.mm(qw_temp, self.A1).view(-1,1,config.NUM_OF_FILTERS)
        g = torch.bmm(g, C)
    
        alph = F.softmax(g, dim=2)
        
        # use attetion on each z get presentation of every review
        alph = alph.view(len(alph), 1, -1)
        d = torch.bmm(C, alph.permute(0, 2, 1))
        return d


class UIEncoder(nn.Module):
    def __init__(self, id_matrix_len):
        super(UIEncoder, self).__init__()
        self.id_embeds = nn.Embedding(id_matrix_len, config.ID_EMBEDDING_DIM)
        self.review_f = config.NUM_OF_FILTERS
        self.l1 = nn.Linear(config.ID_EMBEDDING_DIM, config.ATTEN_VEC_DIM)
        self.A2 = nn.Parameter(torch.randn(config.ATTEN_VEC_DIM, config.NUM_OF_FILTERS), requires_grad=True)
    
    def forward(self, review_matrix, id_num):
        # embeds id_num get id_vector
        # id_num batch_size * 1
        # review_matrix: batch_size * filter * review_size

        # id_vector: batch_size * 1 * id_embed
        id_vector = self.id_embeds(id_num.view(-1)).view(-1, 1, config.ID_EMBEDDING_DIM)

        # MLP get qr
        # qr = batch * 1 * atten_dim
        qr = F.relu(self.l1(id_vector))

        # get attention weight
        # qr: batch * atten_dim
        qr = qr.contiguous().view(-1, config.ATTEN_VEC_DIM)
        
        # e batch * 1 * filter -> batch * 1 * review_size
        e = torch.mm(qr, self.A2).view(-1,1,config.NUM_OF_FILTERS)
        e = torch.bmm(e, review_matrix)

        beta = F.softmax(e, dim=2)

        # use attetion on each d get presentation of every user
        beta = beta.view(len(beta), 1, -1)
        q = torch.bmm(review_matrix, beta.permute(0, 2, 1))

        return q.permute(0, 2, 1)


class FM_Layer(nn.Module):
    def __init__(self):
        super(FM_Layer, self).__init__()
        input_dim = config.NUM_OF_FILTERS*2
        self.linear = nn.Linear(input_dim, 1) 
        
        # bias and weight need to be small, or loss will be big initially 
        self.linear.load_state_dict({"weight": torch.zeros(1, input_dim), "bias": torch.ones(1)}) 
        self.V = nn.Parameter(torch.zeros(input_dim, input_dim))  

    def fm_layer(self, x):
        # linear_part: batch * 1 * input_dim 
        linear_part = self.linear(x)

        batch_size = len(x)
        V = torch.stack((self.V,) * batch_size)

        # batch * 1 * input_dim
        interaction_part_1 = torch.bmm(x, V)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        interaction_part_2 = torch.bmm(torch.pow(x, 2), torch.pow(V, 2))
        list_output = linear_part + torch.sum(0.5 * interaction_part_2 - interaction_part_1)

        rate = torch.stack(tuple(list_output), 0)
        return rate

    def forward(self, x):
        return self.fm_layer(x).view(-1, 1)


class NRPA(nn.Module):

    def __init__(self, weights_matrix_user, weights_matrix_item, uid_matrix_len, iid_matrix_len):
    # Args: 
        super(NRPA, self).__init__()
        self.user_reveiw_net = reviewEncoder(weights_matrix_user, uid_matrix_len)
        self.item_review_net = reviewEncoder(weights_matrix_item, iid_matrix_len)
        self.user_net = UIEncoder(uid_matrix_len)
        self.item_net = UIEncoder(iid_matrix_len)
        self.fm = FM_Layer()

    def forward(self, user_review_vectors, user_id_vector, item_review_vectors, item_id_vector):
        # user_review_vectors: batch * reviews_size * words_size
        batch_size = len(user_id_vector)
        sentence_matrix_user = user_review_vectors.view(-1, config.WORDS_SIZE)
        mul_user_id_vector = torch.cat((user_id_vector,)* config.REVIEWS_SIZE, dim=1).view(-1, 1)
        d_matrix_user = self.user_reveiw_net(sentence_matrix_user, mul_user_id_vector)
        d_matrix_user = d_matrix_user.view(batch_size, config.REVIEWS_SIZE, -1).permute(0,2,1)

        sentence_matrix_item = item_review_vectors.view(-1, config.WORDS_SIZE)
        mul_item_id_vector = torch.cat((item_id_vector,)* config.REVIEWS_SIZE, dim=1).view(-1, 1)
        d_matrix_item = self.item_review_net(sentence_matrix_item, mul_item_id_vector)
        d_matrix_item = d_matrix_item.view(batch_size, config.REVIEWS_SIZE, -1).permute(0,2,1)


        pu = self.user_net(d_matrix_user, user_id_vector)
        qi = self.item_net(d_matrix_item, item_id_vector)

        x = torch.cat((pu, qi), 2)
        rate = self.fm(x)
        return rate


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weight = torch.load('./embed_weight.pkl')
    nrpa = NRPA(weight['weights_matrix_user'], 
                weight['weights_matrix_item'], 
                weight['uid_matrix_len'], 
                weight['iid_matrix_len']).to(device)
    r = torch.randn(20, config.REVIEWS_SIZE, config.WORDS_SIZE, config.WORD_EMBEDDING_DIM).long().to(device)
    u = torch.randn(20, 1).long().to(device)
    rslt = nrpa(r, u, r, u)
    print(rslt.size())
    