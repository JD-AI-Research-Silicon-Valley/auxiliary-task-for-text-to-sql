from __future__ import division
from builtins import bytes
import os
import argparse
import math
import codecs
import torch
import sys
import table
import table.IO
import opts
import random
from itertools import takewhile, count
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
import glob
import json
from tqdm import tqdm
from lib.dbengine import DBEngine
from lib.query import Query

parser = argparse.ArgumentParser(description='evaluate.py')
opts.translate_opts(parser)
opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)
#annotated

if opt.split == 'finaltest':
    split = 'test'
else:
    split = opt.split
if opt.unseen_table!='full':
    opt.anno = os.path.join(
        opt.data_path, 'annotated_ent_'+opt.unseen_table+'/{}.jsonl'.format(split))
    # source
    opt.source_file = os.path.join(
        opt.data_path, 'data_'+opt.unseen_table+'/{}.jsonl'.format(split))
    # DB
    opt.db_file = os.path.join(opt.data_path, 'data/{}.db'.format(split))
else:
    opt.anno = os.path.join(
        opt.data_path, 'annotated_ent/{}.jsonl'.format(split))
    #source
    opt.source_file = os.path.join(
        opt.data_path, 'data/{}.jsonl'.format(split))
    #DB
    opt.db_file = os.path.join(opt.data_path, 'data/{}.db'.format(split))

#pretrained embedding
opt.pre_word_vecs = os.path.join(opt.data_path, 'embedding')


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    opts.train_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    engine = DBEngine(opt.db_file)

    with codecs.open(opt.source_file, "r", "utf-8") as corpus_file:
        sql_list = [json.loads(line)['sql'] for line in corpus_file]

    js_list = table.IO.read_anno_json(opt.anno)

    prev_best = (None, None)
    print(opt.split, opt.model_path)

    num_models=0

    f_out=open('Two-stream-' +opt.unseen_table+'-out-case','w')

    for fn_model in glob.glob(opt.model_path):
        num_models += 1
        sys.stdout.flush()
        print(fn_model)
        print(opt.anno)
        opt.model = fn_model

        translator = table.Translator(opt, dummy_opt.__dict__)
        data = table.IO.TableDataset(js_list, translator.fields, None, False)
        #torch.save(data, open( 'data.pt', 'wb'))
        test_data = table.IO.OrderedIterator(
            dataset=data, device=opt.gpu, batch_size=opt.batch_size, train=False, sort=True, sort_within_batch=False)

        # inference
        r_list = []
        for batch in test_data:
            r_list += translator.translate(batch)
        r_list.sort(key=lambda x: x.idx)
        assert len(r_list) == len(js_list), 'len(r_list) != len(js_list): {} != {}'.format(
            len(r_list), len(js_list))
        # evaluation
        error_cases = []
        for pred, gold, sql_gold in zip(r_list, js_list, sql_list):
            error_cases.append(pred.eval(opt.split, gold, sql_gold, engine))
#            error_cases.append(pred.eval(opt.split, gold, sql_gold))
        print('Results:')
        for metric_name in ('all', 'exe', 'agg', 'sel', 'where', 'col', 'span', 'lay','BIO','BIO_col'):
            c_correct = sum((x.correct[metric_name] for x in r_list))
            print('{}: {} / {} = {:.2%}'.format(metric_name, c_correct,
                                                len(r_list), c_correct / len(r_list)))
            if metric_name=='all':
                all_acc=c_correct
            if metric_name=='exe':
                exe_acc=c_correct
        if prev_best[0] is None or all_acc+exe_acc >prev_best[1]+prev_best[2]:
            prev_best = (fn_model, all_acc, exe_acc)

#        random.shuffle(error_cases)
        for error_case in error_cases:
            if len(error_case) == 0:
                continue
            json.dump(error_case,f_out)
            f_out.write('\n')
#            print('table_id:\t', error_case['table_id'])
#            print('question_id:\t',error_case['question_id'])
#            print('question:\t', error_case['question'])
#            print('table_head:\t', error_case['table_head'])
#            print('table_content:\t', error_case['table_content'])
#            print()

#            print(error_case['BIO'])
#            print(error_case['BIO_col'])
#            print()

#            print('gold:','agg:',error_case['gold']['agg'],'sel:',error_case['predict']['sel'])
#            for i in range(len(error_case['gold']['conds'])):
#                print(error_case['gold']['conds'][i])

 #           print('predict:','agg:',error_case['predict']['agg'],'sel:',error_case['predict']['sel'])
 #           for i in range(len(error_case['predict']['conds'])):
 #               print(error_case['predict']['conds'][i])
 #           print('\n\n')


    print(prev_best)
    if (opt.split == 'dev') and (prev_best[0] is not None) and num_models!=1:
        if opt.unseen_table=='full':
            with codecs.open(os.path.join(opt.save_path, 'dev_best.txt'), 'w', encoding='utf-8') as f_out:
                f_out.write('{}\n'.format(prev_best[0]))
        else:
            with codecs.open(os.path.join(opt.save_path, 'dev_best_'+opt.unseen_table+'.txt'), 'w', encoding='utf-8') as f_out:
                f_out.write('{}\n'.format(prev_best[0]))


if __name__ == "__main__":
    main()
