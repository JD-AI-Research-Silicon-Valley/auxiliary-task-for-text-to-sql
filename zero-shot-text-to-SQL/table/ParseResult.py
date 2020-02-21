import json
from collections import defaultdict
import sys
sys.path.append("..")
from lib.dbengine import DBEngine
from lib.query import Query


class ParseResult(object):
    def __init__(self, idx, agg, sel, cond,BIO,BIO_col):
        self.idx = idx
        self.agg = agg
        self.sel = sel
        self.cond = cond
        self.BIO=BIO
        self.BIO_col=BIO_col
        self.correct = defaultdict(lambda: 0)

    def eval(self,split, gold, sql_gold, engine=None):
        if self.agg == gold['query']['agg']:
            self.correct['agg'] = 1

        if self.sel == gold['query']['sel']:
            self.correct['sel'] = 1



        #gold['BIO_label'].data, gold['BIO_column_label'].data

        op_list_pred = [op for col, op, span in self.cond]
        op_list_gold = [op for col, op, span in gold['query']['conds']]

        col_list_pred = [col for col, op, span in self.cond]
        col_list_gold = [col for col, op, span in gold['query']['conds']]

        q = gold['question']['words']
        span_list_pred = [' '.join(q[span[0]:span[1] + 1])
                          for col, op, span in self.cond]
        span_list_gold = [' '.join(span['words'])
                          for col, op, span in gold['query']['conds']]

        where_pred = list(zip(col_list_pred, op_list_pred, span_list_pred))
        where_gold = list(zip(col_list_gold, op_list_gold, span_list_gold))
        where_pred.sort()
        where_gold.sort()
        if where_pred == where_gold and (len(col_list_pred) == len(col_list_gold)) and (len(op_list_pred) == len(op_list_gold)) and (len(span_list_pred) == len(span_list_gold)):
            self.correct['where'] = 1

        if (len(col_list_pred) == len(col_list_gold)) and ([it[0] for it in where_pred] == [it[0] for it in where_gold]):
            self.correct['col'] = 1

        if (len(op_list_pred) == len(op_list_gold)) and ([it[1] for it in where_pred] == [it[1] for it in where_gold]):
            self.correct['lay'] = 1

        if (len(span_list_pred) == len(span_list_gold)) and ([it[2] for it in where_pred] == [it[2] for it in where_gold]):
            self.correct['span'] = 1

        if all((self.correct[it] == 1 for it in ('agg', 'sel', 'where'))):
            self.correct['all'] = 1

        # execution
        table_id = gold['table_id']
        ans_gold = '0'
        ans_pred = '1'
        if engine is not None:
            ans_gold = engine.execute_query(
                table_id, Query.from_dict(sql_gold), lower=True)

            try:
                sql_pred = {'agg':self.agg, 'sel':self.sel, 'conds': self.recover_cond_to_gloss(gold)}
                ans_pred = engine.execute_query(
                    table_id, Query.from_dict(sql_pred), lower=True)
            except Exception as e:
                ans_pred = repr(e)
        else:
            ans_gold='0'
            ans_pred='1'
        if set(ans_gold) == set(ans_pred):
            self.correct['exe'] = 1

        error_case = {}
        if split == 'finaltest':
            #if self.correct['where'] != 1:
            if True:
                error_case['sel'] = self.correct['sel']
                error_case['where'] = self.correct['where']
                error_case['all'] = self.correct['all']
                error_case['table_id'] = gold['table_id']
                error_case['question_id']=gold['id']
                error_case['question'] = gold['question']['words']
                error_case['table_head'] = [head['words'] for head in gold['table']['header']]
                #error_case['table_content'] = gold['tbl_content']

#                for i in range(len(sql_gold['conds'])):
#                    sql_gold['conds'][i][0] = (
#                    sql_gold['conds'][i][0], gold['table']['header'][sql_gold['conds'][i][0]]['words'],
#                    gold['tbl_content'][sql_gold['conds'][i][0]])
                error_case['gold'] = sql_gold

                error_case['predict'] = {'agg': self.agg.item(), 'sel': self.sel.item(),
                                         'conds': self.print_recover_cond_to_gloss(gold)}
                # error_case['exe result']=self.correct['exe']

                error_case['BIO']=[(x.item(), y) for x, y in zip(list(self.BIO), list(gold['question']['words']))]
                error_case['BIO_col']=self.BIO_col.tolist()

        return error_case


    def recover_cond_to_gloss(self, gold):
        r_list = []
        for col, op, span in self.cond:
            tk_list = []
            for i in range(span[0], span[1] + 1):
                tk_list.append(gold['question']['gloss'][i])
                tk_list.append(gold['question']['after'][i])
            r_list.append([col, op, ''.join(tk_list).strip()])
        return r_list
    def print_recover_cond_to_gloss(self, gold):
        r_list = []
        for col, op, span in self.cond:
            tk_list = []
            for i in range(span[0], span[1] + 1):
                tk_list.append(gold['question']['gloss'][i])
                tk_list.append(gold['question']['after'][i])
            #r_list.append([(col.item(), gold['table']['header'][col.item()]['words'], gold['tbl_content'][col.item()]), op,''.join(tk_list).strip()])
            r_list.append([col.item(), op,''.join(tk_list).strip()])
        return r_list