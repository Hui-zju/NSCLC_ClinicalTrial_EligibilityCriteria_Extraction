import re
from tqdm import tqdm
from IPython.core.debugger import set_trace
import copy
import torch
import torch.nn as nn
import json
from torch.nn.parameter import Parameter
import itertools
from utils.get_bio_tag import process_res_dict
from common.components import HandshakingKernel
import math
import numpy as np


class HandshakingTaggingScheme(object):
    """docstring for HandshakingTaggingScheme"""

    def __init__(self, tok2id, rel2id, max_seq_len):
        super(HandshakingTaggingScheme, self).__init__()
        self.rel2id = rel2id
        self.id2rel = {ind: rel for rel, ind in rel2id.items()}
        self.tok2id = tok2id
        self.id2tok = {ind: tok for tok, ind in tok2id.items()}

        # mapping shaking sequence and matrix
        self.matrix_size = max_seq_len
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_ind2matrix_ind = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in
                                       list(range(self.matrix_size))[ind:]]

        self.matrix_ind2shaking_ind = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_ind2matrix_ind):
            self.matrix_ind2shaking_ind[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def get_spots(self, sample):
        '''
        entity spot and tail_rel spot: (span_pos1, span_pos2, tag_id)
        head_rel spot: (rel_id, span_pos1, span_pos2, tag_id)
        '''

        rel_matrix_spots, head_entity_matrix_spots, tail_entity_matrix_spots = [], [], []
        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            assert (subj_tok_span[1] <= obj_tok_span[0] or obj_tok_span[1] <= subj_tok_span[0])
            rel_catagory = rel['predicate'] if subj_tok_span[0] < obj_tok_span[0] else rel['predicate'] + '_reversed'
            obj_tok_pos_list = [idx for idx in range(obj_tok_span[0], obj_tok_span[1])]
            subj_tok_pos_list = [idx for idx in range(subj_tok_span[0], subj_tok_span[1])]
            if subj_tok_span[0] < obj_tok_span[0]:
                rel_matrix_spots.extend([(subj_tok, obj_tok, self.rel2id.get(rel_catagory, 0)) for obj_tok in obj_tok_pos_list for subj_tok in subj_tok_pos_list])
            else:
                rel_matrix_spots.extend(
                    [(obj_tok, subj_tok, self.rel2id.get(rel_catagory, 0)) for obj_tok in obj_tok_pos_list for subj_tok in
                     subj_tok_pos_list])

        entity_token_tag = []
        for ent in sample["entity_list"]:
            ent_type = ent['type']
            ent_tok_span = ent['tok_span']

            if ent_type.startswith('sent-'):
                assert ent_tok_span[1] - ent_tok_span[0] == 1
                entity_token_tag.append((ent_tok_span[0], self.tok2id[ent['type']]))
            else:
                entity_token_tag.append((ent_tok_span[0], self.tok2id['B-' + ent['type']]))
                for idx in range(ent_tok_span[0] + 1, ent_tok_span[1]):
                    entity_token_tag.append((idx, self.tok2id['I-' + ent['type']]))

        entity_token_tag_combinations = list(itertools.combinations(entity_token_tag, 2))
        head_entity_matrix_spots = [(entity_token_tag_combination[0][0],
                                     entity_token_tag_combination[1][0],
                                     entity_token_tag_combination[0][1])
                                    for entity_token_tag_combination in entity_token_tag_combinations]
        tail_entity_matrix_spots = [(entity_token_tag_combination[0][0],
                                     entity_token_tag_combination[1][0],
                                     entity_token_tag_combination[1][1])
                                    for entity_token_tag_combination in entity_token_tag_combinations]

        return rel_matrix_spots, head_entity_matrix_spots, tail_entity_matrix_spots

    def sharing_spots2shaking_tag(self, spots):
        '''
        convert spots to shaking seq tag
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return:
            shake_seq_tag: (shaking_seq_len, )
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_seq_tag = torch.zeros(shaking_seq_len).long()
        for sp in spots:
            shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
            shaking_seq_tag[shaking_ind] = sp[2]
        return shaking_seq_tag

    def sharing_spots2shaking_tag4batch(self, batch_spots):
        '''
        convert spots to batch shaking seq tag
        因长序列的stack是费时操作，所以写这个函数用作生成批量shaking tag
        如果每个样本生成一条shaking tag再stack，一个32的batch耗时1s，太昂贵
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return:
            batch_shake_seq_tag: (batch_size, shaking_seq_len)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots), shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
                tag_id = sp[2]
                batch_shaking_seq_tag[batch_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag

    def get_sharing_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, )
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_ind in shaking_tag.nonzero():
            shaking_ind_ = shaking_ind[0].item()  # if torch.is_tensor(shaking_ind[0]) else shaking_ind[0]
            tag_id = shaking_tag[shaking_ind_]
            matrix_inds = self.shaking_ind2matrix_ind[shaking_ind_]
            spot = (matrix_inds[0], matrix_inds[1], tag_id.item())
            # if torch.is_tensor(tag_id) else (matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

    def decode_ent_fr_shaking_tag(self,
                                  text,
                                  first_ent_shaking_tag,
                                  second_ent_shaking_tag,
                                  tok2char_span,
                                  tok_offset=0, char_offset=0):
        first_ent_matrix_spots = self.get_sharing_spots_fr_shaking_tag(first_ent_shaking_tag)
        second_ent_matrix_spots = self.get_sharing_spots_fr_shaking_tag(second_ent_shaking_tag)
        token_tag_count_dict = {}
        for matrix_spot in first_ent_matrix_spots:
            if (matrix_spot[0], matrix_spot[2]) not in token_tag_count_dict:
                token_tag_count_dict[(matrix_spot[0], matrix_spot[2])] = 0
            else:
                token_tag_count_dict[(matrix_spot[0], matrix_spot[2])] += 1

        for matrix_spot in second_ent_matrix_spots:
            if (matrix_spot[1], matrix_spot[2]) not in token_tag_count_dict:
                token_tag_count_dict[(matrix_spot[1], matrix_spot[2])] = 0
            else:
                token_tag_count_dict[(matrix_spot[1], matrix_spot[2])] += 1

        token_tag_dict = {}

        token_list = list(set([token_tag_count[0] for token_tag_count in token_tag_count_dict.keys()]))

        # 选取标签数最多的tag作为最终标签
        for token in token_list:
            each_token_tag_count_dict = {key: value for key, value in token_tag_count_dict.items() if key[0] == token}
            tag_id = sorted(each_token_tag_count_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0][1]
            token_tag_dict[token] = tag_id

        tag_list = [0 for _ in range(0, self.matrix_size)]
        for key, value in token_tag_dict.items():
            tag_list[key] = value
        tag_list = [self.id2tok.get(tag, 'O') for tag in tag_list]
        token_pos_start_list = [span[0] for span in tok2char_span if span != (0, 0)]
        token_pos_end_list = [span[1] for span in tok2char_span if span != (0, 0)]
        tokens = [text[span[0]:span[1]] for span in tok2char_span if span != (0, 0)]

        res_dict = {'tokens': tokens, 'label': tag_list,
                    'pos_start': token_pos_start_list, 'pos_end': token_pos_end_list}

        _, entity_pos_start_list, entity_pos_end_list, entity_tag_list = process_res_dict(res_dict)
        entity_list = [text[entity_pos_start_list[idx]:entity_pos_end_list[idx]]
                       for idx in range(0, len(entity_pos_end_list))]
        entity_tok_start_list = [token_pos_start_list.index(pos) for pos in entity_pos_start_list]
        entity_tok_end_list = [token_pos_end_list.index(pos) for pos in entity_pos_end_list]


        entities = [{
            'text': entity_list[idx],
            'type': entity_tag_list[idx],
            'char_span': (entity_pos_start_list[idx] + char_offset, entity_pos_end_list[idx] + char_offset),
            'tok_span': (entity_tok_start_list[idx] + tok_offset, entity_tok_end_list[idx] + tok_offset)
        } for idx in range(0, len(entity_list))]
        return entities


    def decode_rel_fr_shaking_tag(self,
                                  text,
                                  rel_shaking_tag,
                                  first_ent_shaking_tag,
                                  second_ent_shaking_tag,
                                  tok2char_span,
                                  tok_offset=0, char_offset=0):
        '''
        ent shaking tag: (shaking_seq_len, )
        head rel and tail rel shaking_tag: size = (rel_size, shaking_seq_len, )
        '''
        rel_matrix_spots = self.get_sharing_spots_fr_shaking_tag(rel_shaking_tag)
        rel_matrix_spots_dict = {(rel_matrix_spot[0], rel_matrix_spot[1]):rel_matrix_spot[2] for rel_matrix_spot in rel_matrix_spots}
        entity_list = self.decode_ent_fr_shaking_tag(text, first_ent_shaking_tag, second_ent_shaking_tag, tok2char_span)
        entity_index_list = [idx for idx in range(0,len(entity_list))]
        entity_pair_list = list(itertools.combinations(entity_index_list, 2))
        rel_list = []
        for entity_pair in entity_pair_list:
            entity_token_pair_list = [(first_ent_pos, second_ent_pos)
                                      for first_ent_pos in entity_list[entity_pair[0]]['tok_span']
                                      for second_ent_pos in entity_list[entity_pair[1]]['tok_span']]
            entity_token_pair_tag_list = [rel_matrix_spots_dict[entity_token_pair]
                                          for entity_token_pair in entity_token_pair_list
                                          if entity_token_pair in rel_matrix_spots_dict]
            tag = 0 if len(entity_token_pair_tag_list) == 0 else np.argmax(np.bincount(entity_token_pair_tag_list))
            tag = self.id2rel.get(tag, None)
            if tag is not None:
                if tag.endswith('_reversed'):
                    subj = entity_list[entity_pair[1]]
                    obj = entity_list[entity_pair[0]]
                else:
                    subj = entity_list[entity_pair[0]]
                    obj = entity_list[entity_pair[1]]

                rel_list.append({
                    "subject": subj["text"],
                    "object": obj["text"],
                    "subj_tok_span": [subj["tok_span"][0], subj["tok_span"][1]],
                    "obj_tok_span": [obj["tok_span"][0], obj["tok_span"][1]],
                    "subj_char_span": [subj["char_span"][0], subj["char_span"][1]],
                    "obj_char_span": [obj["char_span"][0], obj["char_span"][1]],
                    "predicate": tag.split('_reversed')[0]
                })
        return rel_list


class DataMaker4Bert():
    def __init__(self, tokenizer, handshaking_tagger):
        self.tokenizer = tokenizer
        self.handshaking_tagger = handshaking_tagger

    def get_indexed_data(self, data, max_seq_len, data_type="train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc="Generate indexed train or valid data"):
            text = sample["text"]
            # codes for bert input
            codes = self.tokenizer.encode_plus(text,
                                               return_offsets_mapping=True,
                                               add_special_tokens=False,
                                               max_length=max_seq_len,
                                               truncation=True,
                                               pad_to_max_length=True)

            # tagging
            spots_tuple = None
            if data_type != "test":
                spots_tuple = self.handshaking_tagger.get_spots(sample)

            # get codes
            input_ids = torch.tensor(codes["input_ids"]).long()
            attention_mask = torch.tensor(codes["attention_mask"]).long()
            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            tok2char_span = codes["offset_mapping"]

            sample_tp = (sample,
                         input_ids,
                         attention_mask,
                         token_type_ids,
                         tok2char_span,
                         spots_tuple,
                         )
            indexed_samples.append(sample_tp)
        return indexed_samples

    def generate_batch(self, batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        tok2char_span_list = []

        ent_spots_list = []
        head_rel_spots_list = []
        tail_rel_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])
            token_type_ids_list.append(tp[3])
            tok2char_span_list.append(tp[4])

            if data_type != "test":
                ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = tp[5]
                ent_spots_list.append(ent_matrix_spots)
                head_rel_spots_list.append(head_rel_matrix_spots)
                tail_rel_spots_list.append(tail_rel_matrix_spots)

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)

        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = None, None, None
        if data_type != "test":
            batch_ent_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(ent_spots_list)
            batch_head_rel_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(head_rel_spots_list)
            batch_tail_rel_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(tail_rel_spots_list)

        return sample_list, \
               batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
               batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag


class DataMaker4BiLSTM():
    def __init__(self, text2indices, get_tok2char_span_map, handshaking_tagger):
        self.text2indices = text2indices
        self.handshaking_tagger = handshaking_tagger
        self.get_tok2char_span_map = get_tok2char_span_map

    def get_indexed_data(self, data, max_seq_len, data_type="train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc="Generate indexed train or valid data"):
            text = sample["text"]

            # tagging
            spots_tuple = None
            if data_type != "test":
                spots_tuple = self.handshaking_tagger.get_spots(sample)
            tok2char_span = self.get_tok2char_span_map(text)
            tok2char_span.extend([(-1, -1)] * (max_seq_len - len(tok2char_span)))
            input_ids = self.text2indices(text, max_seq_len)

            sample_tp = (sample,
                         input_ids,
                         tok2char_span,
                         spots_tuple,
                         )
            indexed_samples.append(sample_tp)
        return indexed_samples

    def generate_batch(self, batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        tok2char_span_list = []

        ent_spots_list = []
        head_rel_spots_list = []
        tail_rel_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            tok2char_span_list.append(tp[2])

            if data_type != "test":
                ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = tp[3]
                ent_spots_list.append(ent_matrix_spots)
                head_rel_spots_list.append(head_rel_matrix_spots)
                tail_rel_spots_list.append(tail_rel_matrix_spots)

        batch_input_ids = torch.stack(input_ids_list, dim=0)

        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = None, None, None
        if data_type != "test":
            batch_ent_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(ent_spots_list)
            batch_head_rel_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(head_rel_spots_list)
            batch_tail_rel_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(tail_rel_spots_list)

        return sample_list, \
               batch_input_ids, tok2char_span_list, \
               batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag

class TPLinkerBert(nn.Module):
    def __init__(self, encoder,
                 ent_tag_size,
                 rel_tag_size,
                 shaking_type,
                 inner_enc_type,
                 dist_emb_size,
                 ent_add_dist,
                 rel_add_dist
                 ):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size

        self.ent_fc = nn.Linear(hidden_size, ent_tag_size)
        self.first_ent_fc = nn.Linear(hidden_size, rel_tag_size)
        self.second_ent_fc = nn.Linear(hidden_size, rel_tag_size)

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type, inner_enc_type)

        # distance embedding
        self.dist_emb_size = dist_emb_size
        self.dist_embbedings = None  # it will be set in the first forwarding

        self.ent_add_dist = ent_add_dist
        self.rel_add_dist = rel_add_dist

    def forward(self, input_ids, attention_mask, token_type_ids):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)
        shaking_hiddens4ent = shaking_hiddens
        shaking_hiddens4rel = shaking_hiddens

        # add distance embeddings if it is set
        if self.dist_emb_size != -1:
            # set self.dist_embbedings
            hidden_size = shaking_hiddens.size()[-1]
            if self.dist_embbedings is None:
                dist_emb = torch.zeros([self.dist_emb_size, hidden_size]).to(shaking_hiddens.device)
                for d in range(self.dist_emb_size):
                    for i in range(hidden_size):
                        if i % 2 == 0:
                            dist_emb[d][i] = math.sin(d / 10000 ** (i / hidden_size))
                        else:
                            dist_emb[d][i] = math.cos(d / 10000 ** ((i - 1) / hidden_size))
                seq_len = input_ids.size()[1]
                dist_embbeding_segs = []
                for after_num in range(seq_len, 0, -1):
                    dist_embbeding_segs.append(dist_emb[:after_num, :])
                self.dist_embbedings = torch.cat(dist_embbeding_segs, dim=0)

            if self.ent_add_dist:
                shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)
            if self.rel_add_dist:
                shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)

        rel_shaking_outputs = self.ent_fc(shaking_hiddens4rel)
        first_ent_shaking_outputs = self.first_ent_fc(shaking_hiddens4ent)
        second_ent_shaking_outputs = self.first_ent_fc(shaking_hiddens4ent)

        return rel_shaking_outputs, first_ent_shaking_outputs, second_ent_shaking_outputs


class TPLinkerBiLSTM(nn.Module):
    def __init__(self, init_word_embedding_matrix,
                 emb_dropout_rate,
                 enc_hidden_size,
                 dec_hidden_size,
                 rnn_dropout_rate,
                 rel_size,
                 shaking_type,
                 inner_enc_type,
                 dist_emb_size,
                 ent_add_dist,
                 rel_add_dist):
        super().__init__()
        self.word_embeds = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze=False)
        self.emb_dropout = nn.Dropout(emb_dropout_rate)
        self.enc_lstm = nn.LSTM(init_word_embedding_matrix.size()[-1],
                                enc_hidden_size // 2,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.dec_lstm = nn.LSTM(enc_hidden_size,
                                dec_hidden_size // 2,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.rnn_dropout = nn.Dropout(rnn_dropout_rate)

        hidden_size = dec_hidden_size

        self.ent_fc = nn.Linear(hidden_size, 2)
        self.head_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
        self.tail_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]

        for ind, fc in enumerate(self.head_rel_fc_list):
            self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
        for ind, fc in enumerate(self.tail_rel_fc_list):
            self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type, inner_enc_type)

        # distance embedding
        self.dist_emb_size = dist_emb_size
        self.dist_embbedings = None  # it will be set in the first forwarding

        self.ent_add_dist = ent_add_dist
        self.rel_add_dist = rel_add_dist

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedding: (batch_size, seq_len, emb_dim)
        embedding = self.word_embeds(input_ids)
        embedding = self.emb_dropout(embedding)
        # lstm_outputs: (batch_size, seq_len, enc_hidden_size)
        lstm_outputs, _ = self.enc_lstm(embedding)
        lstm_outputs = self.rnn_dropout(lstm_outputs)
        # lstm_outputs: (batch_size, seq_len, dec_hidden_size)
        lstm_outputs, _ = self.dec_lstm(lstm_outputs)
        lstm_outputs = self.rnn_dropout(lstm_outputs)

        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(lstm_outputs)
        shaking_hiddens4ent = shaking_hiddens
        shaking_hiddens4rel = shaking_hiddens

        # add distance embeddings if it is set
        if self.dist_emb_size != -1:
            # set self.dist_embbedings
            hidden_size = shaking_hiddens.size()[-1]
            if self.dist_embbedings is None:
                dist_emb = torch.zeros([self.dist_emb_size, hidden_size]).to(shaking_hiddens.device)
                for d in range(self.dist_emb_size):
                    for i in range(hidden_size):
                        if i % 2 == 0:
                            dist_emb[d][i] = math.sin(d / 10000 ** (i / hidden_size))
                        else:
                            dist_emb[d][i] = math.cos(d / 10000 ** ((i - 1) / hidden_size))
                seq_len = input_ids.size()[1]
                dist_embbeding_segs = []
                for after_num in range(seq_len, 0, -1):
                    dist_embbeding_segs.append(dist_emb[:after_num, :])
                self.dist_embbedings = torch.cat(dist_embbeding_segs, dim=0)

            if self.ent_add_dist:
                shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)
            if self.rel_add_dist:
                shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)

        ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)

        head_rel_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            head_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        tail_rel_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            tail_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        head_rel_shaking_outputs = torch.stack(head_rel_shaking_outputs_list, dim=1)
        tail_rel_shaking_outputs = torch.stack(tail_rel_shaking_outputs_list, dim=1)

        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs


class MetricsCalculator():
    def __init__(self, handshaking_tagger):
        self.handshaking_tagger = handshaking_tagger

    def get_sample_accuracy(self, pred, truth):
        '''
        计算所有抽取字段都正确的样本比例
        即该batch的输出与truth全等的样本比例
        '''
        # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
        pred_id = torch.argmax(pred, dim=-1)
        # (batch_size, ..., seq_len) -> (batch_size, )，把每个sample压成一条seq
        pred_id = pred_id.view(pred_id.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred_id).float(), dim=1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_)
        return sample_acc

    def get_rel_cpg(self, sample_list, tok2char_span_list,
                    batch_pred_ent_shaking_outputs,
                    batch_pred_head_rel_shaking_outputs,
                    batch_pred_tail_rel_shaking_outputs,
                    pattern="only_head_text"):
        batch_pred_ent_shaking_tag = torch.argmax(batch_pred_ent_shaking_outputs, dim=-1)
        batch_pred_head_rel_shaking_tag = torch.argmax(batch_pred_head_rel_shaking_outputs, dim=-1)
        batch_pred_tail_rel_shaking_tag = torch.argmax(batch_pred_tail_rel_shaking_outputs, dim=-1)

        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_ent_shaking_tag = batch_pred_ent_shaking_tag[ind]
            pred_head_rel_shaking_tag = batch_pred_head_rel_shaking_tag[ind]
            pred_tail_rel_shaking_tag = batch_pred_tail_rel_shaking_tag[ind]

            pred_rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(text,
                                                                              pred_ent_shaking_tag,
                                                                              pred_head_rel_shaking_tag,
                                                                              pred_tail_rel_shaking_tag,
                                                                              tok2char_span)
            gold_rel_list = sample["relation_list"]

            if pattern == "only_head_index":
                gold_rel_set = set(
                    ["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for
                     rel in gold_rel_list])
                pred_rel_set = set(
                    ["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for
                     rel in pred_rel_list])
            elif pattern == "whole_span":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                                rel["subj_tok_span"][1],
                                                                                rel["predicate"],
                                                                                rel["obj_tok_span"][0],
                                                                                rel["obj_tok_span"][1]) for rel in
                                    gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                                rel["subj_tok_span"][1],
                                                                                rel["predicate"],
                                                                                rel["obj_tok_span"][0],
                                                                                rel["obj_tok_span"][1]) for rel in
                                    pred_rel_list])
            elif pattern == "whole_text":
                gold_rel_set = set(
                    ["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in
                     gold_rel_list])
                pred_rel_set = set(
                    ["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in
                     pred_rel_list])
            elif pattern == "only_head_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
                                                                rel["object"].split(" ")[0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
                                                                rel["object"].split(" ")[0]) for rel in pred_rel_list])

            for rel_str in pred_rel_set:
                if rel_str in gold_rel_set:
                    correct_num += 1

            pred_num += len(pred_rel_set)
            gold_num += len(gold_rel_set)

        return correct_num, pred_num, gold_num

    def get_prf_scores(self, correct_num, pred_num, gold_num):
        minimini = 1e-10
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1