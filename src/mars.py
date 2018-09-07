import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
from utils import vocab, pos_vocab, ner_vocab, rel_vocab

class TriAN(nn.Module):

    def __init__(self, args):
        super(TriAN, self).__init__()
        self.args = args
        self.embedding_dim = 300
        self.embedding = nn.Embedding(len(vocab), self.embedding_dim, padding_idx=0)
        self.embedding.weight.data.fill_(0)
        self.embedding.weight.data[:2].normal_(0, 0.1)
        self.pos_embedding = nn.Embedding(len(pos_vocab), args.pos_emb_dim, padding_idx=0)
        self.pos_embedding.weight.data.normal_(0, 0.1)
        self.ner_embedding = nn.Embedding(len(ner_vocab), args.ner_emb_dim, padding_idx=0)
        self.ner_embedding.weight.data.normal_(0, 0.1)
        self.rel_embedding = nn.Embedding(len(rel_vocab), args.rel_emb_dim, padding_idx=0)
        self.rel_embedding.weight.data.normal_(0, 0.1)
        self.RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU}

        self.p_q_emb_match = layers.SeqAttnMatch(self.embedding_dim)

        # Input size to RNN: word emb + question emb + pos emb + ner emb + manual features
        doc_input_size = 2 * self.embedding_dim + args.pos_emb_dim + args.ner_emb_dim + 5 + args.rel_emb_dim

        # Max passage size
        p_max_size = args.p_max_size
        self.p_max_size = p_max_size

        # Max question size
        q_max_size = args.q_max_size
        self.q_max_size = q_max_size

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # RNN question encoder: word emb + pos emb
        qst_input_size = self.embedding_dim + args.pos_emb_dim
        self.question_rnn = layers.StackedBRNN(
            input_size=qst_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        self.doc_hidden_size = doc_hidden_size
        question_hidden_size = 2 * args.hidden_size
        self.question_hidden_size = question_hidden_size
        # print('p_mask : ' , doc_input_size)

        # Attention over passage and question
        self.q_self_attn_start = layers.LinearSeqAttn(question_hidden_size, q_max_size)
        self.p_q_attn_start = layers.BilinearSeqAttn(p_max_size,q_max_size, p_max_size)

        self.q_self_attn_end = layers.LinearSeqAttn(question_hidden_size, q_max_size)
        self.p_q_attn_end = layers.BilinearSeqAttn(p_max_size,q_max_size, p_max_size)

        # Bilinear layer and sigmoid to proba
        self.p_q_bilinear_start = nn.Bilinear(question_hidden_size, question_hidden_size, 1)
        self.p_q_bilinear_end = nn.Bilinear(question_hidden_size, question_hidden_size, 1)
        self.p_linear_start = nn.Linear(question_hidden_size,1)
        self.p_linear_end = nn.Linear(question_hidden_size,1)
        # Attention start end
        self.start_end_attn = layers.BilinearProbaAttn(p_max_size)
        self.end_start_attn = layers.BilinearProbaAttn(p_max_size)

        # Feed forward
        self.feedforward_start = layers.NeuralNet(p_max_size, p_max_size, p_max_size)
        self.feedforward_end = layers.NeuralNet(p_max_size, p_max_size, p_max_size)

    def forward(self, p, p_pos, p_ner, p_mask, q, q_pos, q_mask, f_tensor, p_q_relation):
        p_emb, q_emb = self.embedding(p), self.embedding(q)
        p_pos_emb, p_ner_emb, q_pos_emb = self.pos_embedding(p_pos), self.ner_embedding(p_ner), self.pos_embedding(q_pos)
        p_q_rel_emb = self.rel_embedding(p_q_relation)
        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            p_emb = nn.functional.dropout(p_emb, p=self.args.dropout_emb, training=self.training)
            q_emb = nn.functional.dropout(q_emb, p=self.args.dropout_emb, training=self.training)
            p_pos_emb = nn.functional.dropout(p_pos_emb, p=self.args.dropout_emb, training=self.training)
            p_ner_emb = nn.functional.dropout(p_ner_emb, p=self.args.dropout_emb, training=self.training)
            q_pos_emb = nn.functional.dropout(q_pos_emb, p=self.args.dropout_emb, training=self.training)
            p_q_rel_emb = nn.functional.dropout(p_q_rel_emb, p=self.args.dropout_emb, training=self.training)

        # {WPQi} RNN input part
        p_q_weighted_emb = self.p_q_emb_match(p_emb, q_emb, q_mask)
        p_q_weighted_emb = nn.functional.dropout(p_q_weighted_emb, p=self.args.dropout_emb, training=self.training)

        # Concatenate all features
        p_rnn_input = torch.cat([p_emb, p_q_weighted_emb, p_pos_emb, p_ner_emb, f_tensor, p_q_rel_emb], dim=2)
        q_rnn_input = torch.cat([q_emb, q_pos_emb], dim=2)

        ##### BiLSTM layer
        p_hiddens = self.doc_rnn(p_rnn_input, p_mask)
        # print('p_hiden ', p_hiddens.size())
        q_hiddens = self.question_rnn(q_rnn_input, q_mask)

        #### START ATTENTION LAYER
        q_merge_weights_start = self.q_self_attn_start(q_hiddens)
        q_hidden_start = layers.weighted_avg(q_hiddens, q_merge_weights_start)
        p_merge_weights_start = self.p_q_attn_start(p_hiddens, q_hidden_start)
        # print('p_merge_weights_start', p_merge_weights_start.size())
        p_hidden_start = layers.weighted_avg(q_hidden_start, p_merge_weights_start.transpose(1,2))
        # print('p_hidden_start', p_hidden_start.size())

        #### END ATTENTION LAYER
        q_merge_weights_end = self.q_self_attn_end(q_hiddens)
        # print('q_merge_weights_end', q_merge_weights_end.size())

        q_hidden_end = layers.weighted_avg(q_hiddens, q_merge_weights_end)
        # print('q_hidden_end', q_hidden_end.size())
        p_merge_weights_end = self.p_q_attn_end(p_hiddens, q_hidden_end)
        # print('p_merge_weights_end', q_hidden_end.size())
        p_hidden_end = layers.weighted_avg(q_hidden_end, p_merge_weights_end.transpose(1,2))
        # print('p_hidden_end', p_hidden_start.size())
        #### START SINGLE PROBA MAP

        logits_start = self.p_linear_start(p_hidden_start)
        # print(logits_start.size())


        logits_start.data.masked_fill_(p_mask.unsqueeze(2).data, 0)
        single_map_proba_start = torch.sigmoid(logits_start)
        single_map_proba_start.data.masked_fill_(p_mask.unsqueeze(2).data, 0)

        #### END SINGLE PROBA MAP
        logits_end = self.p_linear_end(p_hidden_end)
        logits_end.data.masked_fill_(p_mask.unsqueeze(2).data, 0)
        single_map_proba_end = torch.sigmoid(logits_end)
        single_map_proba_end.data.masked_fill_(p_mask.unsqueeze(2).data, 0)
        # print('single_map_proba_end ',single_map_proba_end.size())
        #### START END ATTENTION
        attn_map_weights_start = self.start_end_attn(single_map_proba_start, single_map_proba_end, p_mask)
        attn_map_weights_end = self.end_start_attn(single_map_proba_end, single_map_proba_start, p_mask)

        attn_map_start = layers.weighted_avg(single_map_proba_start, attn_map_weights_start)
        attn_map_end = layers.weighted_avg(single_map_proba_end, attn_map_weights_end)

        #### FEED FORWARD
        ff_map_start = self.feedforward_start(attn_map_start.squeeze(2))
        ff_map_end = self.feedforward_end(attn_map_end.squeeze(2))
        ff_map_start.data.masked_fill_(p_mask.data, -float('inf'))
        ff_map_end.data.masked_fill_(p_mask.data, -float('inf'))

        #### OUTPUT
        probas_start = F.softmax(ff_map_start, -1)
        probas_end = F.softmax(ff_map_end, -1)
        print(probas_start, probas_end)
        return probas_start, probas_end
