import re
import torch
import itertools
import numpy as np
from transformers import BertTokenizer
from utils.Classes import Documents, SentenceExtractor, EntityPairsExtractor


def read_annotation(filepath):
    with open(filepath, 'r') as f:
        ann_txt = f.read()
    ann_txt = re.sub('[\n]+', '\n', ann_txt)  # 去除多余的空行
    ann_txt = ann_txt.replace(' ', '')
    ann_txt = ann_txt.replace('\t', '')
    ann_rows = ann_txt.split('\n')

    item_list = ['[entities]', '[events]', '[relations]', '[attributes]']
    sort_index_list = np.argsort([ann_rows.index(item) for item in item_list])
    pos_list = [ann_rows.index(item) for item in item_list] + [len(ann_rows)]
    ent_pos_index = sort_index_list[0]
    ent_rows = ann_rows[pos_list[ent_pos_index]+1:pos_list[ent_pos_index+1]]
    rel_pos_index = sort_index_list[2]
    rel_rows = ann_rows[pos_list[rel_pos_index]+1:pos_list[rel_pos_index+1]]

    rel_rows = [x for x in rel_rows if "<" not in x and "#" not in x and len(x) > 1]
    ent_rows = [x for x in ent_rows if "!" not in x and "#" not in x and len(x) > 1]

    rel_set = []
    for row in rel_rows:
        arg1_pos = row.index('Arg1:')
        arg2_pos = row.index(',Arg2:')
        rel_name = row[0:arg1_pos]
        arg1 = row[arg1_pos + 5:arg2_pos]
        arg2 = row[arg2_pos + 6:]
        arg1_list = arg1.split('|')
        arg2_list = arg2.split('|')
        arg1_arg2 = list(itertools.product(arg1_list, arg2_list))
        rel_list = list(map(lambda x: (x[0], x[1], rel_name), arg1_arg2))
        rel_set.extend(rel_list)
    rel_set = set(rel_set)

    return ent_rows, rel_set


# read the corpus and return them into list of sentences of list of tokens
# read ner training and testing file
def ner_corpus_reader(path, delim='\t', word_idx=0, label_idx=-1):
    tmp_tok, tmp_lab = [], []
    samples = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()  # 去除首尾空格
            cols = line.split(delim)  # 分词
            if len(cols) < 2:  # 遇到空行，就写入一句到tokens和labels
                if len(tmp_tok) > 0:
                    sample = {'tokens': tmp_tok, 'label': tmp_lab}
                    samples.append(sample)
                tmp_tok = []
                tmp_lab = []
            else:
                tmp_tok.append(cols[word_idx])
                tmp_lab.append(cols[label_idx])
    return samples


def re_corpus_reader(path, ann_data_dir, pretrained_model_name_or_path="bert-base-uncased",
             input_method=3, output_method=2, vec_max_len=80, **kwargs):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
    _, rel_set = read_annotation(ann_data_dir)
    rel_name_list = list(set(map(lambda x: x[2], rel_set)))
    rel_name_list.sort()
    rel2idx = dict(zip(rel_name_list, range(1, len(rel_name_list) + 1)))

    docs = Documents(path)
    sent_extractor = SentenceExtractor(sent_split_char='\n', window_size=1, rel_types=rel_set,
                                       filter_no_rel_candidates_sents=True)
    sents = sent_extractor(docs)

    ent_pair_extractor = EntityPairsExtractor(rel_set)
    entity_pairs = ent_pair_extractor(sents)

    doc_ent_pair_ids = {'rel_info': [], 'rel_category': []}
    for doc in docs:
        for rel in doc.rels:
            if rel.category in rel_name_list:
                rel_info = (doc.doc_id, rel.ent1.ent_id, rel.ent2.ent_id)
                rel_category = rel.category
                doc_ent_pair_ids['rel_info'].append(rel_info)
                doc_ent_pair_ids['rel_category'].append(rel_category)

    if len(doc_ent_pair_ids['rel_info']) == 0:
        data = []
        for ent_pair in entity_pairs:
            doc_id = ent_pair.sent.doc_id
            from_end_id = ent_pair.from_ent.ent_id
            to_end_id = ent_pair.to_ent.ent_id
            indexed_tokens, att_mask, pos1, pos2 = bert_tokenize(ent_pair, tokenizer, input_method,
                                                                 output_method, vec_max_len)
            entity_pair = {'indexed_tokens': indexed_tokens, 'att_mask': att_mask,
                           'pos1': pos1, 'pos2': pos2,
                           'doc_id': doc_id, 'from_end_id': from_end_id, 'to_end_id': to_end_id}
            data.append(entity_pair)
        return data

    data = []
    for ent_pair in entity_pairs:
        rel_info = (ent_pair.sent.doc_id, ent_pair.from_ent.ent_id, ent_pair.to_ent.ent_id)
        if rel_info in doc_ent_pair_ids['rel_info']:
            index = doc_ent_pair_ids['rel_info'].index(rel_info)
            label = doc_ent_pair_ids['rel_category'][index]
            label = rel2idx[label]
        else:
            label = 0

        indexed_tokens, att_mask, pos1, pos2 = bert_tokenize(ent_pair, tokenizer, input_method,
                                                             output_method, vec_max_len)
        doc_id = ent_pair.sent.doc_id
        from_end_id = ent_pair.from_ent.ent_id
        to_end_id = ent_pair.to_ent.ent_id
        entity_pair_label = {'indexed_tokens': indexed_tokens,
                             'att_mask': att_mask,
                             'pos1': pos1,
                             'pos2': pos2,
                             'doc_id': doc_id,
                             'from_end_id': from_end_id,
                             'to_end_id': to_end_id,
                             'label': label}
        data.append(entity_pair_label)
    return data


def bert_tokenize(ent_pair, tokenizer, input_method=3, output_method=2, vec_max_len=80):
    sentence = ent_pair.sent.text
    pos_head = [ent_pair.from_ent.start_pos, ent_pair.from_ent.end_pos]
    pos_tail = [ent_pair.to_ent.start_pos, ent_pair.to_ent.end_pos]

    pos_min = pos_head
    pos_max = pos_tail
    if pos_head[0] > pos_tail[0]:
        pos_min = pos_tail
        pos_max = pos_head
        rev = True
    else:
        rev = False

    sen_min = min([ent.start_pos for ent in ent_pair.sent.ents.ents])
    sen_max = max([ent.end_pos for ent in ent_pair.sent.ents.ents])

    # sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
    # ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
    # sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
    # ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
    # sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

    sent0 = tokenizer.tokenize(sentence[sen_min:pos_min[0]])
    ent0 = tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
    sent1 = tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
    ent1 = tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
    sent2 = tokenizer.tokenize(sentence[pos_max[1]:sen_max])

    if input_method == 3:
        ent0 = tokenizer.tokenize(ent_pair.from_ent.category) if not rev else tokenizer.tokenize(
            ent_pair.to_ent.category)
        ent1 = tokenizer.tokenize(ent_pair.to_ent.category) if not rev else tokenizer.tokenize(
            ent_pair.from_ent.category)
        ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
        ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']
        # ent0 = ['#0'] + ent0 + ['#1'] if not rev else ['#2'] + ent0 + ['#3']
        # ent1 = ['#2'] + ent1 + ['#3'] if not rev else ['#0'] + ent1 + ['#1']

    re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
    pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
    pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)

    # make sure pos index are not > max_length
    pos1 = min(vec_max_len - 1, pos1)
    pos2 = min(vec_max_len - 1, pos2)

    indexed_tokens = tokenizer.convert_tokens_to_ids(re_tokens)
    avai_len = len(indexed_tokens)

    # Position
    if output_method == 2 or input_method == 2:  # Mention pooling
        # compute end pos
        if not rev:
            end1 = pos1 + len(ent0) - 1
            end2 = pos2 + len(ent1) - 1
        else:
            end1 = pos1 + len(ent1) - 1
            end2 = pos2 + len(ent0) - 1
        end1 = min(vec_max_len - 1, end1)
        end2 = min(vec_max_len - 1, end2)

        pos1 = torch.tensor([[pos1, end1]]).long()
        pos2 = torch.tensor([[pos2, end2]]).long()
    else:
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

    # Padding
    while len(indexed_tokens) < vec_max_len:
        indexed_tokens.append(0)  # 0 is id for [PAD]
    indexed_tokens = indexed_tokens[:vec_max_len]
    indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

    # Attention mask
    att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
    att_mask[0, :avai_len] = 1

    return indexed_tokens, att_mask, pos1, pos2