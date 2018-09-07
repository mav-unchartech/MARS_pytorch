import torch
import numpy as np
from utils import vocab, pos_vocab, ner_vocab, rel_vocab

class Example:

    def __init__(self, input_dict):
        self.id = input_dict['id']
        self.passage = input_dict['d_words']
        self.question = input_dict['q_words']
        # self.choice = input_dict['c_words']
        self.d_pos = input_dict['d_pos']
        self.d_ner = input_dict['d_ner']
        self.q_pos = input_dict['q_pos']
        assert len(self.q_pos) == len(self.question.split()), (self.q_pos, self.question)
        assert len(self.d_pos) == len(self.passage.split())
        self.features = np.stack([input_dict['in_q'], input_dict['in_c'], \
                                    input_dict['lemma_in_q'], input_dict['lemma_in_c'], \
                                    input_dict['tf']], 1)
        assert len(self.features) == len(self.passage.split())
        self.y_start = input_dict['y_start']
        self.y_end = input_dict['y_end']
        self.y = input_dict['y']
        #self.d_positions = np.array(input_dict['d_positions'])
        self.d_tensor = torch.LongTensor([vocab[w] for w in self.passage.split()])
        self.q_tensor = torch.LongTensor([vocab[w] for w in self.question.split()])
        #self.c_tensor = torch.LongTensor([vocab[w] for w in self.choice.split()])
        self.d_pos_tensor = torch.LongTensor([pos_vocab[w] for w in self.d_pos])
        self.q_pos_tensor = torch.LongTensor([pos_vocab[w] for w in self.q_pos])
        self.d_ner_tensor = torch.LongTensor([ner_vocab[w] for w in self.d_ner])
        self.features = torch.from_numpy(self.features).type(torch.FloatTensor)
        self.p_q_relation = torch.LongTensor([rel_vocab[r] for r in input_dict['p_q_relation']])
        #self.p_c_relation = torch.LongTensor([rel_vocab[r] for r in input_dict['p_c_relation']])

    def __str__(self):
        return 'Passage: %s\n Question: %s\n  Label: %d' % (self.passage, self.question, self.y)

def _to_indices_and_mask(batch_tensor, need_mask=True, is_question=False):
    if is_question: mx_len = 25#args.q_max_size#max([t.size(0) for t in batch_tensor])
    else: mx_len = 750#args.p_max_size
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(0)
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(1)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(0)
    if need_mask:
        return indices, mask
    else:
        return indices

def _to_feature_tensor(features):
    mx_len = 750#max([f.size(0) for f in features])
    batch_size = len(features)
    f_dim = features[0].size(1)
    f_tensor = torch.FloatTensor(batch_size, mx_len, f_dim).fill_(0)
    for i, f in enumerate(features):
        f_tensor[i, :len(f), :].copy_(f)
    return f_tensor

def _out_tensor(features):
    mx_len = 750#max([len(f[0]) for f in features])
    batch_size = len(features)
    f_dim = len(features[0])
    f_tensor = torch.FloatTensor(batch_size, f_dim, mx_len).fill_(0)
    for i, f in enumerate(features):
        f_tensor[i, :,:len(f[0])].copy_(torch.FloatTensor(f))
    return f_tensor

def batchify(batch_data):
    p, p_mask = _to_indices_and_mask([ex.d_tensor for ex in batch_data])
    p_pos = _to_indices_and_mask([ex.d_pos_tensor for ex in batch_data], need_mask=False)
    p_ner = _to_indices_and_mask([ex.d_ner_tensor for ex in batch_data], need_mask=False)
    p_q_relation = _to_indices_and_mask([ex.p_q_relation for ex in batch_data], need_mask=False)
    q, q_mask = _to_indices_and_mask([ex.q_tensor for ex in batch_data], is_question= True)
    q_pos = _to_indices_and_mask([ex.q_pos_tensor for ex in batch_data],is_question= True, need_mask=False)
    f_tensor = _to_feature_tensor([ex.features for ex in batch_data])
    y = _out_tensor([ex.y for ex in batch_data])
    return p, p_pos, p_ner, p_mask, q, q_pos, q_mask, f_tensor, p_q_relation, y
