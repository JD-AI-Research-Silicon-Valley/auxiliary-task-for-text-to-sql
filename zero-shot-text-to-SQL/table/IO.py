# -*- coding: utf-8 -*-

import codecs
import json
import random as rnd
import numpy as np
from collections import Counter, defaultdict
from itertools import chain, count
from six import string_types

import torch
import torchtext.data
import torchtext.vocab
import h5py

UNK_WORD = '<unk>'
UNK = 0
PAD_WORD = '<blank>'
PAD = 1
BOS_WORD = '<s>'
EOS_WORD = '</s>'
SPLIT_WORD = '<|>'
special_token_list = [UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD, SPLIT_WORD]
zero_vector = []
for _ in range(1024):
    zero_vector.append(0.0)

def __getstate__(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def __setstate__(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__


def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                 specials=list(special_token_list),
                                 max_size=vocab_size)


def join_dicts(*args):
    """
    args: dictionaries with disjoint keys
    returns: a single dictionary that has the union of these keys
    """
    return dict(chain(*[d.items() for d in args]))


class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            self.batches = torchtext.data.pool(
                self.data(), self.batch_size,
                self.sort_key, self.batch_size_fn,
                random_shuffler=self.random_shuffler)
        else:
            self.batches = []
            _=0
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))
                _+=1
            print(_,self.batch_size,_*self.batch_size)


#"conds": [ [5, 0, {"words": ["butler", "cc", "-lrb-", "ks", "-rrb-"], "after": [" ", " ", "", "", ""], "gloss": ["Butler", "CC", "(", "KS", ")"]}] ]

def find(q, c):
    ans=[]
    for st in range(0, len(q) - len(c) + 1):
        if q[st:st + len(c)] == c:
            ans.append(st)
            return st
    return 0


def read_anno_json(anno_path):
    with codecs.open(anno_path, "r", "utf-8") as corpus_file:
        js_list = [json.loads(line) for line in corpus_file]
        for js in js_list:
            #cond_list = list(enumerate(js['query']['conds']))
            # sort by (op, orginal index)
            # cond_list.sort(key=lambda x: (x[1][1], x[0]))
            #cond_list.sort(key=lambda x: x[1][1])
            #js['query']['conds'] = [x[1] for x in cond_list]

            cond_list = js['query']['conds']
            pos=[]
            for i in range(len(cond_list)):
                pos.append(find(js['question']['words'],cond_list[i][2]['words']))
            #print(pos)
            S=list(zip(cond_list,pos))
            S.sort(key=lambda x: x[1])
            js['query']['conds'] = [x[0] for x in S]
    return js_list


class TableDataset(torchtext.data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        return -len(ex.src)

    def __init__(self, anno, fields, opt, filter_ex, **kwargs):
        """
        Create a TranslationDataset given paths and fields.

        anno: location of annotated data / js_list
        filter_ex: False - keep all the examples for evaluation (should not have filtered examples); True - filter examples with unmatched spans;
        """
        if isinstance(anno, string_types):
            js_list = read_anno_json(anno)
        else:
            js_list = anno



        src_data = self._read_annotated_file(                               #return a generator
            opt, js_list, 'question', filter_ex)
        src_examples = self._construct_examples(src_data, 'src')            #return a generator of dict ('src':data)

        #elmo_data = self._read_annotated_file(  # return a generator
        #    opt, js_list, 'elmo', filter_ex)
        #elmo_examples = self._construct_examples(elmo_data, 'elmo')  # return a generator of dict ('src':data)


        ent_data = self._read_annotated_file(opt, js_list, 'ent', filter_ex)
        ent_examples = self._construct_examples(ent_data, 'ent')

        type_data = self._read_annotated_file(opt, js_list, 'type', filter_ex)
        type_examples = self._construct_examples(type_data, 'type')

        agg_data = self._read_annotated_file(opt, js_list, 'agg', filter_ex)
        agg_examples = self._construct_examples(agg_data, 'agg')

        sel_data = self._read_annotated_file(opt, js_list, 'sel', filter_ex)
        sel_examples = self._construct_examples(sel_data, 'sel')

        tbl_data = self._read_annotated_file(opt, js_list, 'tbl', filter_ex)
        tbl_examples = self._construct_examples(tbl_data, 'tbl')

        tbl_split_data = self._read_annotated_file(
            opt, js_list, 'tbl_split', filter_ex)
        tbl_split_examples = self._construct_examples(
            tbl_split_data, 'tbl_split')

        tbl_mask_data = self._read_annotated_file(
            opt, js_list, 'tbl_mask', filter_ex)
        tbl_mask_examples = self._construct_examples(
            tbl_mask_data, 'tbl_mask')

        lay_data = self._read_annotated_file(opt, js_list, 'lay', filter_ex)
        lay_examples = self._construct_examples(lay_data, 'lay')

        cond_op_data = self._read_annotated_file(
            opt, js_list, 'cond_op', filter_ex)
        cond_op_examples = self._construct_examples(cond_op_data, 'cond_op')

        cond_col_data = list(
            self._read_annotated_file(opt, js_list, 'cond_col', filter_ex))
        cond_col_examples = self._construct_examples(cond_col_data, 'cond_col')
        cond_col_loss_examples = self._construct_examples(
            cond_col_data, 'cond_col_loss')

        def _map_to_sublist_index(d_list, idx):
            return [([it[idx] for it in d] if (d is not None) else None) for d in d_list]
        span_data = list(self._read_annotated_file(
            opt, js_list, 'cond_span', filter_ex))

        ###
        BIO_label_data = list(self._read_annotated_file(
            opt, js_list, 'BIO_label', filter_ex))

        BIO_label_examples = self._construct_examples( BIO_label_data, 'BIO_label')
        BIO_label_loss_examples = self._construct_examples(BIO_label_data, 'BIO_label_loss')

        ###
        BIO_op_label_data = list(self._read_annotated_file(
            opt, js_list, 'BIO_op_label', filter_ex))

        BIO_op_label_examples = self._construct_examples(BIO_op_label_data, 'BIO_op_label')
        BIO_op_label_loss_examples = self._construct_examples(BIO_op_label_data, 'BIO_op_label_loss')

        ###
        BIO_column_label_data = list(self._read_annotated_file(
            opt, js_list, 'BIO_column_label', filter_ex))
        BIO_column_label_examples = self._construct_examples(BIO_column_label_data, 'BIO_column_label')
        BIO_column_label_loss_examples = self._construct_examples(BIO_column_label_data, 'BIO_column_label_loss')

        #print(len(list(BIO_label_loss_examples)))


        span_l_examples = self._construct_examples(
            _map_to_sublist_index(span_data, 0), 'cond_span_l')
        span_r_examples = self._construct_examples(
            _map_to_sublist_index(span_data, 1), 'cond_span_r')
        span_l_loss_examples = self._construct_examples(
            _map_to_sublist_index(span_data, 0), 'cond_span_l_loss')
        span_r_loss_examples = self._construct_examples(
            _map_to_sublist_index(span_data, 1), 'cond_span_r_loss')





        # examples: one for each src line or (src, tgt) line pair.
        examples = [join_dicts(*it) for it in zip(src_examples, ent_examples, type_examples, agg_examples, sel_examples, lay_examples, tbl_examples, tbl_split_examples, tbl_mask_examples,
                                                  cond_op_examples, cond_col_examples, span_l_examples, span_r_examples, cond_col_loss_examples, span_l_loss_examples, span_r_loss_examples,
                                                  BIO_label_examples, BIO_label_loss_examples,BIO_column_label_examples,BIO_column_label_loss_examples,BIO_op_label_examples,BIO_op_label_loss_examples)]
        # the examples should not contain None
        len_before_filter = len(examples)
        print(len_before_filter)
        examples = list(filter(lambda x: all(
            (v is not None for k, v in x.items())), examples))
        len_after_filter = len(examples)
        print(len_after_filter)
        num_filter = len_before_filter - len_after_filter
        # if num_filter > 0:
        #     print('Filter #examples (with None): {} / {} = {:.2%}'.format(num_filter,
        #                                                                   len_before_filter, num_filter / len_before_filter))

        len_lay_list = []
        len_tgt_list = []
        for ex in examples:
            has_agg = 0 if int(ex['agg']) == 0 else 1
            if len(ex['cond_op']) == 0:
                len_lay_list.append(0)
                len_tgt_list.append(1 + has_agg + 1)
            else:
                len_lay = len(ex['cond_op']) * 2
                len_lay_list.append(len_lay)
                len_tgt_list.append(
                    1 + has_agg + 1 + len_lay + len(ex['cond_op']) * 2)

        # Peek at the first to see which fields are used.
        ex = examples[0]
        keys = ex.keys()
        fields = [(k, fields[k])
                  for k in (list(keys) + ["indices"])]

        def construct_final(examples):
            for i, ex in enumerate(examples):
                yield torchtext.data.Example.fromlist(
                    [ex[k] for k in keys] + [i],
                    fields)

        def filter_pred(example):
            return True

        super(TableDataset, self).__init__(
            construct_final(examples), fields, filter_pred)


    def _read_annotated_file(self, opt, js_list, field, filter_ex):
        """
        path: location of a src or tgt file
        truncate: maximum sequence length (0 for unlimited)
        """
        if field in ('sel', 'agg'):
            lines = (line['query'][field] for line in js_list)
        elif field in ('ent',):
            lines = (line['question']['ent'] for line in js_list)
        elif field in ('type',):
            #lines = (line['question']['word_type'] for line in js_list)

            def filt_type(line):
                return [ w if w =='column' else 'others' for w in line['question']['word_type']]

            lines = (filt_type(line) for line in js_list)
        elif field in ('tbl',):
            def _tbl(line):
                tk_list = [SPLIT_WORD]
                tk_split = '\t' + SPLIT_WORD + '\t'
                tk_list.extend(tk_split.join(
                    ['\t'.join(col['words']) for col in line['table']['header']]).strip().split('\t'))
                tk_list.append(SPLIT_WORD)
                return tk_list
            lines = (_tbl(line) for line in js_list)
        elif field in ('tbl_split',):
            def _cum_length_for_split(line):
                len_list = [len(col['words'])
                            for col in line['table']['header']]
                r = [0]
                for i in range(len(len_list)):
                    r.append(r[-1] + len_list[i] + 1)
                return r
            lines = (_cum_length_for_split(line) for line in js_list)
        elif field in ('tbl_mask',):
            lines = ([0 for col in line['table']['header']]
                     for line in js_list)
        elif field in ('lay',):
            def _lay(where_list):
                return ' '.join([str(op) for col, op, cond in where_list])
            lines = (_lay(line['query']['conds'])
                     for line in js_list)
        elif field in ('cond_op',):
            lines = ([str(op) for col, op, cond in line['query']['conds']]
                     for line in js_list)
        elif field in ('cond_col',):
            lines = ([col for col, op, cond in line['query']['conds']]
                     for line in js_list)
        elif field in ('cond_span'):
            def _find_span(q_list, where_list):
                r_list = []
                for col, op, cond in where_list:
                    tk_list = cond['words']
                    # find exact match first
                    if len(tk_list) <= len(q_list):
                        match_list = []
                        for st in range(0, len(q_list) - len(tk_list) + 1):
                            if q_list[st:st + len(tk_list)] == tk_list:
                                match_list.append((st, st + len(tk_list) - 1))
                        if len(match_list) > 0:
                            r_list.append(rnd.choice(match_list))           #multi match then random choose one.
                            continue
                        elif (opt is not None) and opt.span_exact_match:
                            return None
                        else:
                            # do not have exact match, then fuzzy match (w/o considering order)
                            for len_span in range(len(tk_list), len(tk_list) + 2):
                                for st in range(0, len(q_list) - len_span + 1):
                                    if set(tk_list) <= set(q_list[st:st + len_span]):
                                        match_list.append(
                                            (st, st + len_span - 1))
                                if len(match_list) > 0:
                                    # match spans that are as short as possible
                                    break
                            if len(match_list) > 0:
                                r_list.append(rnd.choice(match_list))
                            else:
                                return None
                    else:
                        return None
                return r_list

            def _span(q_list, where_list, filter_ex):
                r_list = _find_span(q_list, where_list)
                if (not filter_ex) and (r_list is None):
                    r_list = []
                    for col, op, cond in where_list:
                        r_list.append((0, 0))
                return r_list
            lines = (_span(line['question']['words'], line['query']
                           ['conds'], filter_ex) for line in js_list)
            print('span',type(lines))
        elif field in('BIO_label','BIO_column_label','BIO_op_label'):

            def _find_span(q_list, where_list):
                r_list = []
                for col, op, cond in where_list:
                    tk_list = cond['words']
                    # find exact match first
                    if len(tk_list) <= len(q_list):
                        match_list = []
                        for st in range(0, len(q_list) - len(tk_list) + 1):
                            if q_list[st:st + len(tk_list)] == tk_list:
                                match_list.append((st, st + len(tk_list) - 1,col,op))
                        if len(match_list) > 0:
                            r_list.append(rnd.choice(match_list))           #multi match then random choose one.
                            continue
                        elif (opt is not None) and opt.span_exact_match:
                            return None
                        else:
                            # do not have exact match, then fuzzy match (w/o considering order)
                            for len_span in range(len(tk_list), len(tk_list) + 2):
                                for st in range(0, len(q_list) - len_span + 1):
                                    if set(tk_list) <= set(q_list[st:st + len_span]):
                                        match_list.append(
                                            (st, st + len_span - 1,col,op))
                                if len(match_list) > 0:
                                    # match spans that are as short as possible
                                    break
                            if len(match_list) > 0:
                                r_list.append(rnd.choice(match_list))
                            else:
                                return None
                    else:
                        return None
                return r_list

            def _span(q_list, where_list, filter_ex):
                r_list = _find_span(q_list, where_list)
                if (not filter_ex) and (r_list is None):
                    r_list = []
                    for col, op, cond in where_list:
                        r_list.append((0, 0,-1,-1))
                return r_list

            def _col_showin_q(question, cols):  # q: a list of word. col: a list of col, each col is a list of words
                def _find_col(c, cols):
                    for i in range(len(cols)):
                        if c == cols[i]:
                            return i
                    return -1

                mapping = []
                for i in range(len(question)):
                    mapping.append(-1)

                for span_len in reversed(range(10)):
                    for i in range(len(question) - span_len):
                        ff = 1
                        for j in range(i, i + span_len + 1):
                            if mapping[j] != -1:  # belong to a col
                                ff = 0
                                break
                        if ff == 0:
                            continue

                        tag = _find_col(question[i:i + span_len + 1], cols)
                        if tag != -1:
                            for j in range(i, i + span_len + 1):
                                mapping[j] = tag
                return mapping

            lines_1 = [line['question']['words'] for line in js_list]

            lines_2 = (_span(line['question']['words'], line['query']['conds'], filter_ex) for line in js_list)

            lines_3 = [line['question']['word_type'] for line in js_list]
            lines_4 = [_col_showin_q(line['question']['words'], [x['words'] for x in line['table']['header']]) for line
                       in js_list]
            if field in ('BIO_label',):
                total_token = 0
                BI_token = 0
                lines=[]
                for qq,span,col_indicator,col_map in zip(lines_1,lines_2,lines_3,lines_4):
                    q=qq[:]
                    #print(q)
                    for i in range(len(q)):
                        q[i]=2
                        total_token+=1
                    for i in range(len(col_map)):
                        if col_map[i] != -1:
                            q[i]=3
#                    for i in range(len(col_indicator)):
#                        if col_indicator[i]=='column':
#                            q[i]=3
                    if span is not None and len(span) > 0:
                        for sp in span:
                            BI_token+=1
                            #print(sp)
                            q[sp[0]]=0
                            for i in range(sp[0]+1,sp[1]+1):
                                q[i]=1
                    #yield q
                    lines.append(q)
                print(total_token, BI_token, 1.0 * BI_token / total_token)
            elif field in ('BIO_column_label'):#BIO_column_label
                lines = []
                for qq, span in zip(lines_1, lines_2):
                    q = qq[:]
                    for i in range(len(q)):
                        q[i]=-1
                    if span is not None and len(span) > 0:
                        for sp in span:
                            for i in range(sp[0],sp[1]+1):
                                q[i]=sp[2]     #cond_col
                    lines.append(q)
            else:
                lines = []
                for qq, span in zip(lines_1, lines_2):
                    q = qq[:]
                    for i in range(len(q)):
                        q[i] = -1
                    if span is not None and len(span) > 0:
                        for sp in span:
                            for i in range(sp[0], sp[1] + 1):
                                q[i] = sp[3]    #cond_op
                    lines.append(q)
            lines = (line for line in lines)

            #print('BIO', type(lines))
        elif field in ('cond_mask',):
            lines = ([0 for col, op, cond in line['query']['conds']]
                     for line in js_list)
        elif field in ('elmo',):
            lines = (line[field]['words'] for line in js_list)
        else:
            lines = (line[field]['words'] for line in js_list)
            #print(field)
            #print(type(js_list))
            #print(type(lines))
        for line in lines:
            yield line

    def _construct_examples(self, lines, side):
        for words in lines:
            example_dict = {side: words}
            yield example_dict

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(TableDataset, self).__reduce_ex__()

    @staticmethod
    def load_fields(vocab):
        vocab = dict(vocab)
        fields = TableDataset.get_fields()
        for k, v in vocab.items():
            # Hack. Can't pickle defaultdict :(
            v.stoi = defaultdict(lambda: 0, v.stoi)
            fields[k].vocab = v
        return fields

    @staticmethod
    def save_vocab(fields):
        vocab = []
        for k, f in fields.items():
            if 'vocab' in f.__dict__:
                f.vocab.stoi = dict(f.vocab.stoi)
                vocab.append((k, f.vocab))
        return vocab

    @staticmethod
    def get_fields():
        fields = {}
        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD, include_lengths=True)#, eos_token=EOS_WORD)



        fields["ent"] = torchtext.data.Field(
            pad_token=PAD_WORD, include_lengths=False)#, eos_token=EOS_WORD)
        fields["type"] = torchtext.data.Field(
            pad_token=PAD_WORD, include_lengths=False)  # , eos_token=EOS_WORD)
        fields["agg"] = torchtext.data.Field(
            sequential=False, use_vocab=False, batch_first=True)
        fields["sel"] = torchtext.data.Field(
            sequential=False, use_vocab=False, batch_first=True)
        fields["tbl"] = torchtext.data.Field(
            pad_token=PAD_WORD, include_lengths=True)
        fields["tbl_split"] = torchtext.data.Field(
            use_vocab=False, pad_token=0)
        fields["tbl_mask"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.ByteTensor, batch_first=True, pad_token=1)
        fields["lay"] = torchtext.data.Field(
            sequential=False, batch_first=True)
        fields["cond_op"] = torchtext.data.Field(
            include_lengths=True, pad_token=PAD_WORD)
        fields["cond_col"] = torchtext.data.Field(
            use_vocab=False, include_lengths=False, pad_token=0)
        fields["cond_span_l"] = torchtext.data.Field(
            use_vocab=False, include_lengths=False, pad_token=0)
        fields["cond_span_r"] = torchtext.data.Field(
            use_vocab=False, include_lengths=False, pad_token=0)
        fields["cond_col_loss"] = torchtext.data.Field(
            use_vocab=False, include_lengths=False, pad_token=-1)
        fields["cond_span_l_loss"] = torchtext.data.Field(
            use_vocab=False, include_lengths=False, pad_token=-1)
        fields["cond_span_r_loss"] = torchtext.data.Field(
            use_vocab=False, include_lengths=False, pad_token=-1)
        fields["indices"] = torchtext.data.Field(
            use_vocab=False, sequential=False)

        fields["BIO_label"] = torchtext.data.Field(
            use_vocab=False, include_lengths=False, pad_token=-1)
        fields["BIO_label_loss"] = torchtext.data.Field(
            use_vocab=False, include_lengths=False, pad_token=-1)
        fields["BIO_column_label"] = torchtext.data.Field(
            use_vocab=False, include_lengths=False, pad_token=-1)
        fields["BIO_column_label_loss"] = torchtext.data.Field(
            use_vocab=False, include_lengths=False, pad_token=-1)

        fields["BIO_op_label"] = torchtext.data.Field(
            use_vocab=False, include_lengths=False, pad_token=-1)
        fields["BIO_op_label_loss"] = torchtext.data.Field(
            use_vocab=False, include_lengths=False, pad_token=-1)

        return fields

    @staticmethod
    def build_vocab(train, dev, test, opt):
        fields = train.fields

        merge_list = []
        merge_name_list = ['tbl','src']#('src', 'tbl')
        print(1)
        for split in (dev, test, train,):

            for merge_name_it in merge_name_list:
                fields[merge_name_it].build_vocab(split, max_size=opt.src_vocab_size, min_freq=0)
                merge_list.append(fields[merge_name_it].vocab)
                print(merge_name_it, len(fields[merge_name_it].vocab.stoi))
        print(2)
        # build vocabulary only based on the training set
        fields["ent"].build_vocab(
            train, max_size=opt.src_vocab_size, min_freq=0)
        fields["type"].build_vocab(
            train, max_size=opt.src_vocab_size, min_freq=0)
        fields["lay"].build_vocab(
            train, max_size=opt.src_vocab_size, min_freq=0)
        fields["cond_op"].build_vocab(
            train, max_size=opt.src_vocab_size, min_freq=0)
        print(3)
        # need to know all the words to filter the pretrained word embeddings
        merged_vocab = merge_vocabs(merge_list, vocab_size=opt.src_vocab_size)
        total=0
        for x in merge_list:
            total+=len(x.stoi)
        print(total,len(merged_vocab.stoi))
        for merge_name_it in merge_name_list:
            fields[merge_name_it].vocab = merged_vocab
