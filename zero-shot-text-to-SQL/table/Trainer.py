"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
"""
from __future__ import division
import os
import time
import sys
import math
import torch
import torch.nn as nn

import table
import table.modules
from table.Utils import argmax
import random

class Statistics(object):
    def __init__(self, loss, eval_result):
        self.loss = loss
        self.eval_result = eval_result
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        for k, v in stat.eval_result.items():
            if k in self.eval_result:
                v0 = self.eval_result[k][0] + v[0]
                v1 = self.eval_result[k][1] + v[1]
                self.eval_result[k] = (v0, v1)
            else:
                self.eval_result[k] = (v[0], v[1])

    def accuracy(self, return_str=False):
        d = sorted([(k, v)
                    for k, v in self.eval_result.items()], key=lambda x: x[0])
        #print(d)
        if return_str:
            return '; '.join((('{}: {:.2%}'.format(k, 1.0*v[0] / v[1])) for k, v in d))
        else:
            return dict([(k, 100.0 * v[0] / v[1]) for k, v in d])

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        print(("Epoch %2d, %5d/%5d; %s; %.0f s elapsed") %
              (epoch, batch, n_batches, self.accuracy(True), time.time() - start))
        sys.stdout.flush()

    def log(self, split, logger, lr, step):
        pass


def count_accuracy(scores, target, mask=None, row=False):
    pred = argmax(scores)
    if mask is None:
        m_correct = pred.eq(target)
        num_all = m_correct.numel()
    elif row:
        m_correct = pred.eq(target).masked_fill_(
            mask.type(torch.bool), 1).prod(0, keepdim=False)

        #print('m_correct_row', m_correct.type())
        num_all = m_correct.numel()
    else:
        non_mask = mask.ne(1).type(torch.bool)
        m_correct = pred.eq(target).masked_select(non_mask)
        num_all = non_mask.sum().item()

    m_correct = m_correct.type(torch.LongTensor)
    if torch.cuda.is_available():
        m_correct=m_correct.cuda()
    return (m_correct, num_all)



def count_condition_value_F1(scores1,golden_scores1, target_ls1,target_rs1):
    preds = argmax(scores1)

    preds = preds.transpose(0,1)
    target_ls=target_ls1.transpose(0,1)
    target_rs=target_rs1.transpose(0,1)
    golden_scores=golden_scores1.transpose(0,1)

   # print(type(preds),preds.size(),target_ls.size())

    total_p=0
    total_r=0
    matched = 0
    exact_matched=0
    for sample_id in range(preds.size(0)):
        pred=preds[sample_id]
        #print(pred)
        golden_score=golden_scores[sample_id]
        #print(g)
        target_l=target_ls[sample_id]
        #print(target_l)
        target_r=target_rs[sample_id]
        #print(target_r)
        exact_matched+=1
        for i in range(pred.size(0)):
            if pred[i]!=golden_score[i] and golden_score[i]!=-1:
                exact_matched-=1
                break
        cond_span_lr = []
        l=0
        r=0
        for i in range(pred.size(0)):
            if pred[i]==0:
                if l!=0:
                    cond_span_lr.append((l,r))
                l=i
                r=i
            elif pred[i]==1:
                r=i
            else:
                if l!=0:
                    cond_span_lr.append((l,r))
                l=0
                r=0
        if l != 0:
            cond_span_lr.append((l, r))

        for l,r in cond_span_lr:
            for i in range(target_l.size(0)):
                if l==target_l[i] and r==target_r[i]:
                    matched+=1
        for i in range(target_l.size(0)):
            if target_l[i]!=-1:
                total_r+=1
        total_p+=len(cond_span_lr)

    #print(matched,total_p,total_r)
    #if random.random()<0.01:
    #    print(preds[:10])
    #    print(golden_scores[:10])
    recall=1.0*matched/(total_r+1e-10)
    precision=1.0*matched/(total_p+1e-10)
    return (exact_matched,preds.size(0)),(precision,1),(recall,1),(recall*precision*2,(recall+precision+1e-10))
    #for span in target:
    #    if span in cond_span_lr:
    #        recall+=1
    #for span in cond_span_lr:
    #    if span in target:
    #        precision+=1
    #precision=len(cond_span_lr)>0 ? 1.0*precision/len(cond_span_lr) : 0
    #recall = len(target) > 0 ? 1.0 * recall / len(stargetpan): 0
    #return precision*recall/(precision+recall)/2.0



def count_condition_value_EM_column_op(scores1,scores_col1,scores_op1, golden_scores1, golden_scores_col1,golden_scores_op1, target_ls1, target_rs1, target_cols1):
    preds = argmax(scores1)
    preds_col = argmax(scores_col1)
    preds_op = argmax(scores_op1)

    preds = preds.transpose(0,1)
    preds_col = preds_col.transpose(0, 1)
    preds_op = preds_op.transpose(0, 1)

    target_ls = target_ls1.transpose(0,1)
    target_rs = target_rs1.transpose(0,1)
    golden_scores = golden_scores1.transpose(0,1)
    golden_scores_col = golden_scores_col1.transpose(0, 1)
    golden_scores_op = golden_scores_op1.transpose(0, 1)
    target_cols = target_cols1.transpose(0,1)

    total_p=0
    total_r=0

    exact_matched=0
    exact_matched_op=0
    exact_matched_col=0
    for sample_id in range(preds.size(0)):
        pred=preds[sample_id]
        pred_col=preds_col[sample_id]
        pred_op = preds_op[sample_id]
        golden_score=golden_scores[sample_id]
        golden_score_col = golden_scores_col[sample_id]
        golden_score_op = golden_scores_op[sample_id]

        exact_matched += 1
        exact_matched_op += 1
        exact_matched_col += 1
        BIO_not_match = False
        for i in range(pred.size(0)):
            if pred[i] != golden_score[i] and golden_score[i] != -1:
                exact_matched-=1
                exact_matched_op-=1
                exact_matched_col-=1
                BIO_not_match = True
                break

        if BIO_not_match == False:
            col_not_match = False
            for i in range(pred.size(0)):
                if pred[i]==0:
                    column_cnt = []
                    for j in range(torch.max(pred_col) + 2):
                        column_cnt.append(0)
                    column_cnt[pred_col[i]] = 1
                    for j in range(i+1,pred.size(0)):
                        if pred[j]!=1:
                            break
                        column_cnt[pred_col[j]] += 1
                    max_cnt=0
                    argmax1=pred_col[i]
                    for j in range(torch.max(pred_col)+2):
                        if column_cnt[j]>max_cnt:
                            max_cnt=column_cnt[j]
                            argmax1=j
                    if argmax1!=golden_score_col[i]:
                        exact_matched_col-=1
                        col_not_match=True
                        break

            op_not_match = False
            for i in range(pred.size(0)):
                if pred[i]==0:
                    op_cnt = []
                    for j in range(3):
                        op_cnt.append(0)
                    op_cnt[pred_op[i]] = 1
                    for j in range(i+1,pred.size(0)):
                        if pred[j]!=1:
                            break
                        op_cnt[pred_op[j]] += 1
                    max_cnt=0
                    argmax1=pred_op[i]
                    for j in range(3):
                        if op_cnt[j]>max_cnt:
                            max_cnt=op_cnt[j]
                            argmax1=j
                    if argmax1!=golden_score_op[i]:
                        exact_matched_op-=1
                        op_not_match=True
                        break
            if op_not_match or col_not_match:
                exact_matched-=1

    return (exact_matched,preds.size(0)),(exact_matched_col,preds.size(0)),(exact_matched_op,preds.size(0)),0



def count_condition_value_F1_column(scores1,scores_col1, golden_scores1, golden_scores_col1, target_ls1, target_rs1,gold_cols):
    preds = argmax(scores1)
    pred_cols = argmax(scores_col1)

    preds = preds.transpose(0, 1)
    target_ls = target_ls1.transpose(0, 1)
    target_rs = target_rs1.transpose(0, 1)

    golden_scores = golden_scores1.transpose(0, 1)
    pred_cols = pred_cols.transpose(0,1)
    golden_scores_col1 = golden_scores_col1.transpose(0,1)
    gold_cols = gold_cols.transpose(0, 1)
    # print(type(preds),preds.size(),target_ls.size())

    total_p = 0
    total_r = 0
    matched = 0
    exact_matched = 0
    for sample_id in range(preds.size(0)):
        pred = preds[sample_id]
        # print(pred)
        golden_score = golden_scores[sample_id]
        # print(g)
        target_l = target_ls[sample_id]
        # print(target_l)
        target_r = target_rs[sample_id]
        # print(target_r)

        pred_col=pred_cols[sample_id]
        golden_score_col1 = golden_scores_col1[sample_id]
        gold_col = gold_cols[sample_id]

        exact_matched += 1
        for i in range(pred.size(0)):
            if pred[i] != golden_score[i] and golden_score[i] != -1:
                exact_matched -= 1
                break
        cond_span_lr = []
        l = 0
        r = 0
        for i in range(pred.size(0)):
            if pred[i] == 0:
                if l != 0:
                    cond_span_lr.append((l, r))
                l = i
                r = i
            elif pred[i] == 1:
                r = i
            else:
                if l != 0:
                    cond_span_lr.append((l, r))
                l = 0
                r = 0
        if l != 0:
            cond_span_lr.append((l, r))

        for l, r in cond_span_lr:
            for i in range(target_l.size(0)):
                #print(l,r,pred_col.size(),gold_col.size(),target_l.size(0))
                if l == target_l[i] and r == target_r[i] and pred_col[l]==gold_col[i]:
                    matched += 1
        for i in range(target_l.size(0)):
            if target_l[i] != -1:
                total_r += 1
        total_p += len(cond_span_lr)

    # print(matched,total_p,total_r)
    # if random.random()<0.01:
    #    print(preds[:10])
    #    print(golden_scores[:10])
    recall = 1.0 * matched / (total_r + 1e-10)
    precision = 1.0 * matched / (total_p + 1e-10)
    return (exact_matched, preds.size(0)), (precision, 1), (recall, 1), (recall * precision * 2, (recall + precision + 1e-10))


def count_where_accuracy(score_span_l, score_span_r, score_col, gold_span_ls, gold_span_rs, gold_cols):
    pred_span_ls = argmax(score_span_l)
    pred_span_rs = argmax(score_span_r)
    pred_cols = argmax(score_col)
    pred_span_ls =pred_span_ls.transpose(0,1)
    pred_span_rs = pred_span_rs.transpose(0, 1)
    pred_cols = pred_cols.transpose(0,1)
    gold_span_ls = gold_span_ls.transpose(0, 1)
    gold_span_rs = gold_span_rs.transpose(0, 1)
    gold_cols = gold_cols.transpose(0, 1)
    exact_matched = 0
    for sample_id in range(pred_span_ls.size(0)):
        pred_span_l = pred_span_ls[sample_id]
        pred_span_r = pred_span_rs[sample_id]
        pred_col = pred_cols[sample_id]

        gold_span_l = gold_span_ls[sample_id]
        gold_span_r = gold_span_rs[sample_id]
        gold_col = gold_cols[sample_id]

        exact_matched+=1
        for i in range(pred_span_l.size(0)):
            if gold_col[i]!=-1:
                if gold_col[i]!=pred_col[i] or gold_span_l[i]!=pred_span_l[i] or gold_span_r[i]!= pred_span_r[i]:
                    exact_matched-=1
                    break
    return exact_matched,pred_span_ls.size(0)

def aggregate_accuracy(r_dict, metric_name_list):
    m_list = []
    for metric_name in metric_name_list:
        m_list.append(r_dict[metric_name][0])
        #print(r_dict[metric_name][0].size(),r_dict[metric_name][0].type())
    #print (len(m_list),m_list[0].type(),m_list[0].size())
    agg= torch.stack(m_list, dim=0)
    agg = agg.prod(0, keepdim=False)
    return (agg.sum().item(), agg.numel())


class Trainer(object):
    def __init__(self, model, train_iter, valid_iter,
                 train_loss, valid_loss, optim):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            valid_iter: the validate data iterator.
            train_loss: the train side LossCompute object for computing loss.
            valid_loss: the valid side LossCompute object for computing loss.
            optim: the optimizer responsible for lr update.
        """
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim

        # Set model in training mode.
        self.model.train()

    def forward(self, batch, criterion):
        # 1. F-prop.
        q, q_len = batch.src
        #print(q)
        tbl, tbl_len = batch.tbl
        cond_op, cond_op_len = batch.cond_op
        agg_out, sel_out, lay_out, cond_col_out, cond_span_l_out, cond_span_r_out, BIO_out, BIO_column_out, BIO_op_out = self.model(
            q, q_len, batch.ent,batch.type, tbl, tbl_len, batch.tbl_split, batch.tbl_mask, cond_op, cond_op_len, batch.cond_col, batch.cond_span_l, batch.cond_span_r, batch.lay)

        # 2. Compute loss.
        pred = {'agg': agg_out, 'sel': sel_out, 'lay': lay_out, 'cond_col': cond_col_out,
                'cond_span_l': cond_span_l_out, 'cond_span_r': cond_span_r_out, 'BIO_label': BIO_out,
                'BIO_column_label': BIO_column_out, 'BIO_op_label':BIO_op_out}
        #print(lay_out)
        gold = {'agg': batch.agg, 'sel': batch.sel, 'lay': batch.lay, 'cond_col': batch.cond_col_loss,
                'cond_span_l': batch.cond_span_l_loss, 'cond_span_r': batch.cond_span_r_loss,'BIO_label': batch.BIO_label_loss, 'BIO_column_label': batch.BIO_column_label_loss,
                'BIO_op_label':batch.BIO_op_label_loss}
        #print(batch.lay)
        loss = criterion.compute_loss(pred, gold)

        # 3. Get the batch statistics.
        r_dict = {}
        #print('1',argmax(pred['agg'].data))
        #print('2',gold['agg'].data)

        for metric_name in ('agg', 'sel', 'lay'):
            r_dict[metric_name] = count_accuracy(
                pred[metric_name].data, gold[metric_name].data)
        for metric_name in ('cond_col', 'cond_span_l', 'cond_span_r'):
            #r_dict[metric_name + '-token'] = count_accuracy(
            #    pred[metric_name].data, gold[metric_name].data, mask=gold[metric_name].data.eq(-1), row=False)
            r_dict[metric_name] = count_accuracy(
                pred[metric_name].data, gold[metric_name].data, mask=gold[metric_name].data.eq(-1), row=True)

        for metric_name in ('BIO_label','BIO_column_label','BIO_op_label'):
            r_dict[metric_name] = count_accuracy(
                pred[metric_name].data, gold[metric_name].data, mask=gold[metric_name].data.eq(-1), row=False)

        #print('3', r_dict['agg'][0])
        st = dict([(k, (int(v[0].sum()), v[1])) for k, v in r_dict.items()])
        #print('4', st['agg'])
        prf=count_condition_value_F1(pred['BIO_label'].data, gold['BIO_label'].data,gold['cond_span_l'].data,gold['cond_span_r'].data)
        st['BIO_label-EM'] = prf[0]
        #st['BIO_label-P'] = prf[1]
        #st['BIO_label-R'] = prf[2]
        st['BIO_label-F1']=prf[3]


        prf = count_condition_value_EM_column_op(pred['BIO_label'].data, pred['BIO_column_label'].data, pred['BIO_op_label'].data,
                                              gold['BIO_label'].data, gold['BIO_column_label'].data,gold['BIO_op_label'].data,
                                              gold['cond_span_l'].data,
                                              gold['cond_span_r'].data, gold['cond_col'].data)

        st['ALL-label-EM'] = prf[0]
        st['BIO_coloum_label-EM'] = prf[1]
        st['BIO_op_label-EM'] = prf[2]

        prf = count_condition_value_F1_column(pred['BIO_label'].data, pred['BIO_column_label'].data,
                                                 gold['BIO_label'].data, gold['BIO_column_label'].data,
                                                 gold['cond_span_l'].data,
                                                 gold['cond_span_r'].data,gold['cond_col'].data)
        st['BIO_coloum_label-P'] = prf[1]
        st['BIO_coloum_label-R'] = prf[2]
        st['BIO_coloum_label-F1'] = prf[3]


        #st['where'] = aggregate_accuracy(
        #    r_dict, ('lay', 'cond_col', 'cond_span_l', 'cond_span_r'))
        st['where'] = count_where_accuracy(
            pred['cond_span_l'].data, pred['cond_span_r'].data, pred['cond_col'].data,
            gold['cond_span_l'].data, gold['cond_span_r'].data, gold['cond_col'].data
        )


        st['all'] = aggregate_accuracy(
            r_dict, ('agg', 'sel', 'lay', 'cond_col', 'cond_span_l', 'cond_span_r'))
        #st['all'] = aggregate_accuracy(
        #    r_dict, ('agg', 'sel', 'where'))
        #batch_stats = Statistics(loss.data[0], st)
        batch_stats = Statistics(loss.data.item(), st)

        return loss, batch_stats

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics(0, {})
        report_stats = Statistics(0, {})
        #print(type(self.train_iter))
        for i, batch in enumerate(self.train_iter):
            self.model.zero_grad()
            #print(batch)
            loss, batch_stats = self.forward(batch, self.train_loss)

            # Update the parameters and statistics.
            loss.backward()
            self.optim.step()
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            if report_func is not None:
                report_stats = report_func(
                    epoch, i, len(self.train_iter),
                    total_stats.start_time, self.optim.lr, report_stats)

        return total_stats

    def validate(self):
        """ Called for each epoch to validate. """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics(0, {})
        for batch in self.valid_iter:
            loss, batch_stats = self.forward(batch, self.valid_loss)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, eval_metric, epoch):
        """ Called for each epoch to update learning rate. """
        return self.optim.updateLearningRate(eval_metric, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Called conditionally each epoch to save a snapshot. """

        model_state_dict = self.model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        checkpoint = {
            'model': model_state_dict,
            'vocab': table.IO.TableDataset.save_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim
        }
        eval_result = valid_stats.accuracy()
        torch.save(checkpoint, os.path.join(
            opt.save_path, 'm_%d.pt' % (epoch)))
