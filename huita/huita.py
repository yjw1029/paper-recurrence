import numpy as np
import torch
from torch import nn
import torch.nn.functional as F 
import config



class SentenceEncoder(nn.Module):
    def __init__(self, word_weight_matrix):
        super(SentenceEncoder, self).__init__()
        word_num_embeddings, word_embedding_dim = word_weight_matrix.size()
        self.word_embeds = nn.Embedding(word_num_embeddings, word_embedding_dim)
        self.word_embeds.load_state_dict({'weight': word_weight_matrix})

        self.conv = nn.Conv1d(
            in_channels = config.WORD_EMBEDDING_DIM,
            out_channels = config.WORD_LEVEL_FILTERS, 
            kernel_size = config.WINDOW_SIZE_CNN,
            padding = (config.WINDOW_SIZE_CNN -1) //2
            )
        self.V = nn.Parameter(torch.zeros(1, config.WORD_LEVEL_FILTERS))
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, sentence_vector):
        # sentence_vector batch * word_size
        sentence_vector = sentence_vector.view(-1)
        sentence_matrix = self.word_embeds(sentence_vector).view(-1, config.WORDS_SIZE, config.WORD_EMBEDDING_DIM)
        sentence_matrix = sentence_matrix.permute(0, 2, 1)

        # sentence_matrix batch * word_embed_dim * word_size
        # C batch * filter * word_size
        C = F.relu(self.conv(sentence_matrix))
        batch_size = len(C)

        # trans V into batch size
        # V_batch batch * 1 * filters
        # alph: batch * 1 * wordsize
        V_batch = torch.stack((self.V,)*batch_size, dim=0)
        alph = F.softmax(torch.tanh(torch.bmm(V_batch, C) + self.b), dim=2)

        # output = batch * word_level_filter * 1 --> batch * word_filter
        output = torch.bmm(C, alph.permute(0,2,1)).view(-1, config.WORD_LEVEL_FILTERS)
        return output


class reviewEncoder(nn.Module):
    def __init__(self):
        super(reviewEncoder, self).__init__()
        self.conv = nn.Conv1d(
            in_channels = config.WORD_LEVEL_FILTERS,
            out_channels = config.SEN_LEVEL_FILTERS, 
            kernel_size = config.WINDOW_SIZE_CNN,
            padding = (config.WINDOW_SIZE_CNN -1) // 2
            )
        self.V = nn.Parameter(torch.zeros(1, config.SEN_LEVEL_FILTERS))
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, review_matrix):
        # review_matrix: batch * WORD_LEVEL_FILTERS * sentence_size
        # C: batch * SEN_LEVEL_FILTERS * sentence_size
        C = F.relu(self.conv(review_matrix))
        batch_size = len(C)

        # trans V into batch size
        # alpha: batch * 1 * sentence_size
        V_batch = torch.stack((self.V,)*batch_size, dim=0)
        alph = F.softmax(torch.tanh(torch.bmm(V_batch, C) + self.b), dim=2)

        # output: batch * SEN_LEVEL_FILTERS
        output = torch.bmm(C, alph.permute(0,2,1)).view(-1, config.SEN_LEVEL_FILTERS)
        return output


class UIEncoder(nn.Module):
    def __init__(self, id_matrix_len):
        super(UIEncoder, self).__init__()
        self.V = nn.Parameter(torch.zeros(1, config.SEN_LEVEL_FILTERS))
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.id_embeds = nn.Embedding(id_matrix_len, config.ID_EMBEDDING_DIM)
        
    def forward(self, all_reviews_vector, id_num):
        # all_reviews_vector: batch * sen_filter * review_size
        # id_num: batch * 1  ;id_vector: batch * 1 * id_embed_size
        id_vec = self.id_embeds(id_num.view(-1)).view(-1, 1, config.ID_EMBEDDING_DIM)
        batch_size = len(all_reviews_vector)

        # trans V into batch size
        V_batch = torch.stack((self.V,)*batch_size, dim=0)
        alph = F.softmax(torch.tanh(torch.bmm(V_batch, all_reviews_vector) + self.b), dim=0)

        # r: batch * 1 * sen_filter
        r = torch.bmm(all_reviews_vector, alph.permute(0,2,1)).permute(0,2,1)

        # output: batch * 1 * (sen_filter + d_embed_size)
        output = torch.cat((r,id_vec), dim=2)
        return output


class RatingPrediction(nn.Module):
    def __init__(self):
        super(RatingPrediction, self).__init__()
        self.linear = nn.Linear(config.ID_EMBEDDING_DIM + config.SEN_LEVEL_FILTERS, 1)
        self.linear.load_state_dict({"weight": torch.zeros(1, config.ID_EMBEDDING_DIM+config.SEN_LEVEL_FILTERS), "bias": torch.ones(1)})
        
    def forward(self, u, t):
        # u: batch * 1 * (SEN_LEVEL_FILTERS + ID_EMBEDDING_DIM)
        x = torch.mul(u,t)
        output = F.relu(self.linear(x))
        return output.view(-1, 1)


class HUITA(nn.Module):
    # weights_matrix_user, weights_matrix_item, uid_matrix_len, iid_matrix_len
    def __init__(self, weights_matrix_user, weights_matrix_item, uid_matrix_len, iid_matrix_len):
        super(HUITA, self).__init__()
        self.sentence_en_user = SentenceEncoder(weights_matrix_user)
        self.sentence_en_item = SentenceEncoder(weights_matrix_item)
        self.review_en_user = reviewEncoder()
        self.review_en_item = reviewEncoder()
        self.user_en = UIEncoder(uid_matrix_len)
        self.item_en = UIEncoder(iid_matrix_len)
        self.rate_pre = RatingPrediction()

    def forward(self, all_user_reviews, all_item_reviews, user_id_num, item_id_num):
        # all_xx_reviews: batch * review_size * sentence_size * word_size
        # xx_id_num: batch * 1

        # XX_sentences_vec = (batch * review_size * sentence_size) * word_size
        user_sentences_vec = all_user_reviews.view(-1, config.WORDS_SIZE)
        item_sentences_vec = all_item_reviews.view(-1, config.WORDS_SIZE)
        
        # review_matrix: (batch * review_size * sentence_size) * word_level_filter
        # -> (batch * review_size) * sentence_size * word_level_filter
        user_review_matrix = self.sentence_en_user(user_sentences_vec)
        user_review_matrix = user_review_matrix.view(-1, config.SENTENCES_SIZE, config.WORD_LEVEL_FILTERS)
        item_review_matrix = self.sentence_en_item(item_sentences_vec)
        item_review_matrix = item_review_matrix.view(-1, config.SENTENCES_SIZE, config.WORD_LEVEL_FILTERS)

        # user_all_review_matrix (batch * review_size) * sentence_filter
        # -> batch * review_size * sentence_filter
        user_allreview_matrix = self.review_en_user(user_review_matrix.permute(0, 2, 1))
        user_allreview_matrix = user_allreview_matrix.view(-1, config.REVIEWS_SIZE, config.SEN_LEVEL_FILTERS)
        item_allreview_matrix = self.review_en_item(item_review_matrix.permute(0, 2, 1))
        item_allreview_matrix = item_allreview_matrix.view(-1, config.REVIEWS_SIZE, config.SEN_LEVEL_FILTERS)


        user_present = self.user_en(user_allreview_matrix.permute(0, 2, 1), user_id_num)
        item_present = self.item_en(item_allreview_matrix.permute(0, 2, 1), item_id_num)

        predict_rate =  self.rate_pre(user_present, item_present)
        return predict_rate

