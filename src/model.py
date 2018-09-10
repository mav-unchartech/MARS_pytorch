import logging
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import math
from utils import vocab
from doc import batchify
from mars import TriAN

logger = logging.getLogger()

class Model:

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.finetune_topk = args.finetune_topk
        self.lr = args.lr
        self.p_max_size= args.p_max_size
        self.use_cuda = (args.use_cuda == True) and torch.cuda.is_available()
        print('Use cuda:', self.use_cuda)
        if self.use_cuda:
            torch.cuda.set_device(int(args.gpu))
        self.network = TriAN(args)
        self.init_optimizer()

        if args.pretrained:
            print('Load pretrained model from %s...' % args.pretrained)
            self.load(args.pretrained)
        else:
            self.load_embeddings(vocab.tokens(), args.embedding_file)
        self.network.register_buffer('fixed_embedding', self.network.embedding.weight.data[self.finetune_topk:].clone())
        if self.use_cuda:
            self.network.cuda()

        self._report_num_trainable_parameters()

    def _report_num_trainable_parameters(self):
        num_parameters = 0
        for p in self.network.parameters():
            if p.requires_grad:
                sz = list(p.size())
                if sz[0] == len(vocab):
                    sz[0] = self.finetune_topk
                num_parameters += np.prod(sz)
        print('Number of parameters: ', num_parameters)

    def train(self, train_data):
        self.network.train()
        self.updates = 0
        iter_cnt, num_iter = 0, (len(train_data) + self.batch_size - 1) // self.batch_size
        for batch_input in self._iter_data(train_data):
            feed_input = [x for x in batch_input[:-1]]
            y = batch_input[-1]
            y = y.transpose(0,1)
            y_start = y[0]
            y_end = y[1]
            y_end_liste = []
            y_start_liste = []
            to_remove = []
            for i in range(len(y_end)):
                try:
                    a = list(y_start[i].numpy()).index(1)
                    b = list(y_end[i].numpy()).index(1)
                except ValueError:
                    to_remove.append(i)
                    print('There is an question without answer')
                    continue
                y_start_liste.append(a)
                y_end_liste.append(b)

            y_start = torch.LongTensor(y_start_liste)
            y_end = torch.LongTensor(y_end_liste)

            #y_end = [int(i) for i in batch_input[-1][1]]
            #y_start = [int(i) for i in batch_input[-1][0]]
            pred_proba = self.network(*feed_input)
            pred_proba_start = pred_proba[0]
            pred_proba_end = pred_proba[1]

            m = nn.LogSoftmax()
            # for i in to_remove:
            #     pred_proba_start = pred_proba_start.detach().numpy()
            #     pred_proba_end = pred_proba_end.detach().numpy()
            #     pred_proba_start = np.delete(pred_proba_start,i, axis = 0)
            #     pred_proba_end = np.delete(pred_proba_end,i, axis = 0)
            #     pred_proba_start = torch.LongTensor(pred_proba_start)
            #     pred_proba_end = torch.LongTensor(pred_proba_end)
            # pred_proba_start = torch.Tensor.numpy(pred_proba_start.data)[0]
            # pred_proba_end = torch.Tensor.numpy(pred_proba_end.data)[0]


            # loss = F.binary_cross_entropy(pred_proba, y) #/!\FLAG
            #loss = -(math.log(pred_proba_start[y1])+math.log(pred_proba_start[y2]))  size_average=True
            loss1 = F.nll_loss(m(pred_proba_start), y_start, size_average=True)
            loss2 = F.nll_loss(m(pred_proba_end), y_end, size_average=True)
            loss = (loss1 + loss2) / 2
            print(loss)
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.grad_clipping)

            # Update parameters
            self.optimizer.step()
            self.network.embedding.weight.data[self.finetune_topk:] = self.network.fixed_embedding
            self.updates += 1
            iter_cnt += 1

            if self.updates % 20 == 0:
                print('loss.data :',loss.data)
                print('loss item ', loss.item())
                print('Iter: %d/%d, Loss: %f' % (iter_cnt, num_iter, loss.item()))
        self.scheduler.step()
        print('LR:', self.scheduler.get_lr()[0])

    def output_to_map(prediction):
        maps = []
        for sub_start, sub_end in zip(prediction[0], prediction[1]):
            start = int(torch.argmax(sub_start))
            end = int(torch.argmax(sub_end))
            start, end = min(start, end), max(start, end)
            maps.append([0] * start + [1] * (end - start) + [0]*(len(sub_start) - end))
        return maps

    def map_padding(start, end):
        padded = []
        padded.extend([0] * start.index(1))
        padded.extend([1] * (end.index(1) - start.index(1)))
        padded.extend([0] * (len(start) - end.index(1)))
        return padded

    def evaluate(self, dev_data):
        from sklearn.metrics import f1_score
        f1_scores = []
        for batch_input in self._iter_data(dev_data):
            y = batch_input[-1]
            y = y.transpose(0,1)
            feed_input = [x for x in batch_input[:-1]]
            pred_proba = self.network(*feed_input)
            y_start = y[0]
            y_end = y[1]
            y_end_liste = []
            y_start_liste = []
            to_remove = []
            for i in range(len(y_end)):
                try:
                    a = list(y_start[i].numpy()).index(1)
                    b = list(y_end[i].numpy()).index(1)
                except ValueError:
                    to_remove.append(i)
                    print('There is an question without answer')
                    continue
                y_start_liste.append(a)
                y_end_liste.append(b)

            pred_proba_start = pred_proba[0]
            pred_proba_end = pred_proba[1]
            for i in range(len(y_end)):
                predictions = [0] * self.p_max_size
                answer = [0] * self.p_max_size
                for j in range(y_start_liste[i],y_end_liste[i]):
                    answer[j] = 1
                max_index_start = np.argmax(pred_proba_start[i].detach().numpy())
                max_index_end = np.argmax(pred_proba_end[i].detach().numpy())
                # print(max_index_start)
                # print(max_index_end)
                if max_index_start <= max_index_end :
                    for j in range(max_index_start, max_index_end+1):
                        predictions[j] = 1
                # print('answer ', answer)
                # print('predictions ', predictions)
                # print('f1_score : ',f1_score(answer, predictions))
                f1_scores.append(f1_score(answer, predictions))
        return sum(f1_scores)/len(f1_scores)

        #     map_pred = self.output_to_map(pred_proba)
        #     for i, data in enumerate(feed_input):
        #         truth = self.map_padding(data.y_start, data.y_end)
        #         f1_scores.append(f1_score(truth, map_pred[i][:len(truth)]))
        # return sum(f1_scores)/len(f1_scores)


    def predict(self, test_data):
        # DO NOT SHUFFLE test_data
        self.network.eval()
        prediction = []
        for batch_input in self._iter_data(test_data):
            feed_input = [x for x in batch_input[:-1]]
            # print('batch input ',batch_input[-1])

            # y = [int(i) for i in batch_input[-1][0]] #not using the last input: y
            pred_proba = self.network(*feed_input)
            pred_proba_start = pred_proba[0]
            pred_proba_end = pred_proba[1]
            pred_proba_start = pred_proba_start.data.cpu()
            pred_proba_end = pred_proba_end.data.cpu()

            #### prendre le max des probas
            prediction += [pred_proba_start,pred_proba_end]
        return prediction

    def _iter_data(self, data):
        num_iter = (len(data) + self.batch_size - 1) // self.batch_size
        for i in range(num_iter):
            start_idx = i * self.batch_size
            batch_data = data[start_idx:(start_idx + self.batch_size)]
            if len(batch_data) < self.batch_size:
                batch_data += (self.batch_size - len(batch_data)) * [batch_data[-1]]
            batch_input = batchify(batch_data)

            # Transfer to GPU
            if self.use_cuda:
                batch_input = [Variable(x.cuda(async=True)) for x in batch_input]
            else:
                batch_input = [Variable(x) for x in batch_input]
            yield batch_input

    def load_embeddings(self, words, ile):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            ile: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in vocab}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), ile))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(ile) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = vocab.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[vocab[w]].copy_(vec)
                    else:
                        logging.warning('WARN: Duplicate embedding found for %s' % w)
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[vocab[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[vocab[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        print('parameters ', len(parameters))
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.lr,
                                       momentum=0.4,
                                       weight_decay=0)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                        lr=self.lr,
                                        weight_decay=0)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 15], gamma=0.5)

    def save(self, ckt_path):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {'state_dict': state_dict}
        torch.save(params, ckt_path)

    def load(self, ckt_path):
        logger.info('Loading model %s' % ckt_path)
        saved_params = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        return self.network.load_state_dict(state_dict, strict=False)

    def cuda(self):
        self.use_cuda = True
        self.network.cuda()
