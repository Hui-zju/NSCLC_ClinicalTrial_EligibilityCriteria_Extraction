import os
import re
import numpy as np
from itertools import permutations, chain


class Entity(object):  # 实体id、类别、起始位置、结束位置、文本；offset函数对起始位置、结束位置加了偏移，并重新构建了一个对象Entity
    def __init__(self, ent_id, category, start_pos, end_pos, text):
        self.ent_id = ent_id
        self.category = category
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.text = text

    def __gt__(self, other):
        return self.start_pos > other.start_pos

    def offset(self, offset_val):
        return Entity(self.ent_id,
                      self.category,
                      self.start_pos + offset_val,
                      self.end_pos + offset_val,
                      self.text)

    def __repr__(self):
        fmt = '({ent_id}, {category}, ({start_pos}, {end_pos}), {text})'
        return fmt.format(**self.__dict__)


class Entities(object):  # 实体列表，构建了一个实体的词典；offset偏移；
    def __init__(self, ents):
        self.ents = sorted(ents)
        self.ent_dict = dict(zip([ent.ent_id for ent in ents], ents))

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            return self.ents[key]
        else:
            return self.ent_dict.get(key, None)

    def __len__(self):
        return len(self.ents)

    def offset(self, offset_val):
        ents = [ent.offset(offset_val) for ent in self.ents]
        return Entities(ents)

    def vectorize(self, vec_len, cate2idx):  # 将实体位置标记为类型id，其他位置为0
        res_vec = np.zeros(vec_len, dtype=int)
        for ent in self.ents:
            res_vec[ent.start_pos: ent.end_pos] = cate2idx[ent.category]
        return res_vec

    def find_entities(self, start_pos, end_pos):  # 重新构造实体？
        res = []
        for ent in self.ents:
            if ent.start_pos > end_pos:
                break
            sp, ep = (max(start_pos, ent.start_pos), min(end_pos, ent.end_pos))
            if ep > sp:
                new_ent = Entity(ent.ent_id, ent.category, sp, ep, ent.text[:(ep - sp)])
                res.append(new_ent)
        return Entities(res)

    def __add__(self, other):  # 增加实体
        ents = self.ents + other.ents
        return Entities(ents)

    def merge(self):  #  融合实体？
        merged_ents = []
        for ent in self.ents:
            if len(merged_ents) == 0:
                merged_ents.append(ent)
            elif (merged_ents[-1].end_pos == ent.start_pos and
                  merged_ents[-1].category == ent.category):
                merged_ent = Entity(ent_id=merged_ents[-1].ent_id,
                                    category=ent.category,
                                    start_pos=merged_ents[-1].start_pos,
                                    end_pos=ent.end_pos,
                                    text=merged_ents[-1].text + ent.text)
                merged_ents[-1] = merged_ent
            else:
                merged_ents.append(ent)
        return Entities(merged_ents)


class Relation(object):  # 关系id、种类、实体1、实体2
    def __init__(self, rel_id, category, ent1, ent2):
        self.rel_id = rel_id
        self.category = category
        self.ent1 = ent1
        self.ent2 = ent2

    @property
    def is_valid(self):  # 是否是真的关系
        return (isinstance(self.ent1, Entity) and
                isinstance(self.ent2, Entity) and
                [self.ent1.category, self.ent2.category] == re.split('[-_]', self.category))

    @property
    def start_pos(self):  # 关系的开始位置
        return min(self.ent1.start_pos, self.ent2.start_pos)

    @property
    def end_pos(self):  # 关系的结束位置
        return max(self.ent1.end_pos, self.ent2.end_pos)

    def offset(self, offset_val):
        return Relation(self.rel_id,
                        self.category,
                        self.ent1.offset(offset_val),
                        self.ent2.offset(offset_val))

    def __gt__(self, other_rel):
        return self.ent1.start_pos > other_rel.ent1.start_pos

    def __repr__(self):
        fmt = '({rel_id}, {category} Arg1:{ent1} Arg2:{ent2})'
        return fmt.format(**self.__dict__)


class Relations(object):  # 关系列表，
    def __init__(self, rels):
        self.rels = rels

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.rels[key]
        elif isinstance(key, slice):
            return Relations(self.rels[key])

    def __add__(self, other):
        rels = self.rels + other.rels
        return Relations(rels)

    def find_relations(self, start_pos, end_pos):  # 找关系
        res = []
        for rel in self.rels:
            if start_pos <= rel.start_pos and end_pos >= rel.end_pos:
                res.append(rel)
        return Relations(res)

    def offset(self, offset_val):
        return Relations([rel.offset(offset_val) for rel in self.rels])

    @property
    def start_pos(self):
        return min([rel.start_pos for rel in self.rels])

    @property
    def end_pos(self):
        return max([rel.end_pos for rel in self.rels])

    def __len__(self):
        return len(self.rels)

    def __repr__(self):
        return self.rels.__repr__()


class TextSpan(object):  # 文本、实体列表、关系列表
    def __init__(self, text, ents, rels, **kwargs):
        self.text = text
        self.ents = ents
        self.rels = rels

    def __getitem__(self, key):
        if isinstance(key, int):
            start, stop = key, key + 1
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self.text)
        else:
            raise ValueError('parameter should be int or slice')
        if start < 0:
            start += len(self.text)
        if stop < 0:
            stop += len(self.text)
        text = self.text[key]
        ents = self.ents.find_entities(start, stop).offset(-start)
        rels = self.rels.find_relations(start, stop).offset(-start)
        return TextSpan(text, ents, rels)

    def __len__(self):
        return len(self.text)


class Sentence(object):  # 句子类，包含doc_id,offset,textspan(两种输入方式)
    def __init__(self, doc_id, offset, text='', ents=[], rels=[], textspan=None):
        self.doc_id = doc_id
        self.offset = offset
        if isinstance(textspan, TextSpan):
            self.textspan = textspan
        else:
            self.textspan = TextSpan(text, ents, rels)

    @property
    def text(self):
        return self.textspan.text

    @property
    def ents(self):
        return self.textspan.ents

    @property
    def rels(self):
        return self.textspan.rels

    def abbreviate(self, max_len, ellipse_chars='$$'):  #
        if max_len <= len(ellipse_chars):
            return ''
        left_trim = (max_len - len(ellipse_chars)) // 2
        right_trim = max_len - len(ellipse_chars) - left_trim
        return self[:left_trim] + ellipse_chars + self[-right_trim:]

    def __getitem__(self, key):
        if isinstance(key, int):
            start, stop = key, key + 1
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self.text)
        else:
            raise ValueError('parameter should be int or slice')
        if start < 0:
            start += len(self.text)
        if stop < 0:
            stop += len(self.text)
        offset = self.offset + start
        textspan = self.textspan[start: stop]
        return Sentence(self.doc_id, offset, textspan=textspan)

    def __gt__(self, other):
        return self.offset > other.offse

    def __add__(self, other):
        if isinstance(other, str):
            return Sentence(doc_id=self.doc_id, offset=self.offset, text=self.text + other,
                            ents=self.ents, rels=self.rels)
        assert self.doc_id == other.doc_id, 'sentences should be from the same document'
        assert self.offset + len(self) <= other.offset, 'sentences should not have overlap'
        doc_id = self.doc_id
        text = self.text + other.text
        offset = self.offset
        ents = self.ents + other.ents.offset(len(self.text))
        rels = self.rels + other.rels.offset(len(self.text))
        return Sentence(doc_id=doc_id, offset=offset, text=text, ents=ents, rels=rels)

    def __len__(self):
        return len(self.textspan)


class Document(object):  # 文档类，doc_id和textspan
    def __init__(self, doc_id, text, ents, rels):
        self.doc_id = doc_id
        self.textspan = TextSpan(text, ents, rels)

    @property
    def text(self):
        return self.textspan.text

    @property
    def ents(self):
        return self.textspan.ents

    @property
    def rels(self):
        return self.textspan.rels


class Documents(object):  # 文档目录，文档id列表
    def __init__(self, data_dir, doc_ids=None):
        if os.path.isdir(data_dir):
            self.data_dir = data_dir
            self.doc_ids = doc_ids
            if self.doc_ids is None:
                self.doc_ids = self.scan_doc_ids()
        elif os.path.isfile(data_dir):
            doc_id = os.path.split(data_dir)[-1].split('.')[0]
            self.data_dir = os.path.split(data_dir)[0]
            self.doc_ids = [doc_id]
        else:
            raise ValueError('data_dir error')



    def scan_doc_ids(self):  #
        doc_ids = [fname.split('.')[0] for fname in os.listdir(self.data_dir)]
        doc_ids = [doc_id for doc_id in doc_ids if len(doc_id) > 0]
        doc_ids = [doc_id for doc_id in doc_ids if len(self.read_txt_file(doc_id)) > 0]
        return np.unique(doc_ids)

    def read_txt_file(self, doc_id):
        fname = os.path.join(self.data_dir, doc_id + '.txt')
        with open(fname, encoding='utf-8') as f:
            text = f.read()
        return text

    def parse_attribute_line(self, raw_str):
        attr_id, label = raw_str.strip().split('\t')
        category, ent_id, text = label.split(' ')
        attr = (attr_id, category, ent_id, text)
        return attr

    def parse_entity_line(self, raw_str, attrs):
        ent_id, label, text = raw_str.strip().split('\t')
        category, pos = label.split(' ', 1)
        pos = pos.split(' ')
        if attrs is not None:
            for attr in attrs:
                if ent_id == attr[2]:
                    category = attr[3] + '_' + category
        ent = Entity(ent_id, category, int(pos[0]), int(pos[-1]), text)
        return ent

    def parse_relation_line(self, raw_str, ents):
        rel_id, label = raw_str.strip().split('\t')
        category, arg1, arg2 = label.split(' ')
        arg1 = arg1.split(':')[1]
        arg2 = arg2.split(':')[1]
        ent1 = ents[arg1]
        ent2 = ents[arg2]
        return Relation(rel_id, category, ent1, ent2)

    def read_anno_file(self, doc_id, add_attr=False):
        ents = []
        attrs = []
        rels = []
        fname = os.path.join(self.data_dir, doc_id + '.ann')
        with open(fname, encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('A'):
                attr = self.parse_attribute_line(line)
                attrs.append(attr)

        for line in lines:
            if line.startswith('T'):
                if not add_attr:
                    attrs = None
                ent = self.parse_entity_line(line, attrs)
                ents.append(ent)
        ents = Entities(ents)

        for line in lines:
            if line.startswith('R'):
                rel = self.parse_relation_line(line, ents)
                #  if rel.is_valid:
                rels.append(rel)
        rels = Relations(rels)
        return ents, rels

    def __len__(self):
        return len(self.doc_ids)

    def get_doc(self, doc_id):
        text = self.read_txt_file(doc_id)
        ents, rels = self.read_anno_file(doc_id)
        doc = Document(doc_id, text, ents, rels)
        return doc

    def __getitem__(self, key):
        if isinstance(key, int):
            doc_id = self.doc_ids[key]
            return self.get_doc(doc_id)
        if isinstance(key, str):
            doc_id = key
            return self.get_doc(doc_id)
        if isinstance(key, np.ndarray) and key.dtype == int:
            doc_ids = self.doc_ids[key]
            return Documents(self.data_dir, doc_ids=doc_ids)


class SentenceExtractor(object):  # 句子提取类
    def __init__(self, sent_split_char, window_size, rel_types, filter_no_rel_candidates_sents=True):
        self.sent_split_char = sent_split_char
        self.window_size = window_size
        self.filter_no_rel_candidates_sents = filter_no_rel_candidates_sents
        self.rels_type_set = set(map(lambda x: (x[0], x[1]), list(rel_types)))

    def get_sent_boundaries(self, text):
        dot_indices = []
        for i, ch in enumerate(text):
            if ch == self.sent_split_char:
                dot_indices.append(i + 1)

        if len(dot_indices) <= self.window_size - 1:
            return [(0, len(text))]

        dot_indices = [0] + dot_indices
        if text[-1] != self.sent_split_char:
            dot_indices += [len(text)]

        boundries = []
        for i in range(len(dot_indices) - self.window_size):
            start_stop = (
                dot_indices[i],
                dot_indices[i + self.window_size]
            )
            boundries.append(start_stop)
        return boundries

    def has_rels_candidates(self, ents):
        ent_cates = [ent.category for ent in ents]
        for pos_rel in set(permutations(ent_cates, 2)):
            if pos_rel in self.rels_type_set:
                return True
        return False

    def extract_doc(self, doc):
        sents = []
        for start_pos, end_pos in self.get_sent_boundaries(doc.text):
            ents = []
            sent_text = doc.text[start_pos: end_pos]
            for ent in doc.ents.find_entities(start_pos=start_pos, end_pos=end_pos):
                ents.append(ent.offset(-start_pos))
            if self.filter_no_rel_candidates_sents and not self.has_rels_candidates(ents):
                continue
            rels = []
            for rel in doc.rels.find_relations(start_pos=start_pos, end_pos=end_pos):
                rels.append(rel.offset(-start_pos))
            sent = Sentence(doc.doc_id,
                            offset=start_pos,
                            text=sent_text,
                            ents=Entities(ents),
                            rels=Relations(rels))
            sents.append(sent)
        return sents

    def __call__(self, docs):
        sents = []
        for doc in docs:
            sents += self.extract_doc(doc)
        return sents


class EntityPair(object):  # 实体对类
    def __init__(self, doc_id, sent, from_ent, to_ent):
        self.doc_id = doc_id
        self.sent = sent
        self.from_ent = from_ent
        self.to_ent = to_ent

    def __repr__(self):
        fmt = 'doc {}, sent {}, {} -> {}'
        return fmt.format(self.doc_id, self.sent.text, self.from_ent, self.to_ent)


class EntityPairsExtractor(object):
    def __init__(self, allow_rel_types, max_len=5000, ellipse_chars='$$', pad=1000):
        self.allow_rel_types = set(map(lambda x: (x[0], x[1]), list(allow_rel_types)))
        self.max_len = max_len
        self.pad = pad
        self.ellipse_chars = ellipse_chars

    def extract_candidate_rels(self, sent):
        candidate_rels = []
        for f_ent, t_ent in permutations(sent.ents, 2):
            rel_cate = (f_ent.category, t_ent.category)
            if rel_cate in self.allow_rel_types:
                candidate_rels.append((f_ent, t_ent))
        return candidate_rels

    def make_entity_pair(self, sent, f_ent, t_ent):
        doc_id = sent.doc_id
        if f_ent.start_pos < t_ent.start_pos:
            left_ent, right_ent = f_ent, t_ent
        else:
            left_ent, right_ent = t_ent, f_ent
        start_pos = max(0, left_ent.start_pos - self.pad)
        end_pos = min(len(sent), right_ent.end_pos + self.pad)
        res_sent = sent[start_pos: end_pos]

        # if len(res_sent) > self.max_len:
        #     res_sent = res_sent.abbreviate(self.max_len)
        f_ent = res_sent.ents[f_ent.ent_id]
        t_ent = res_sent.ents[t_ent.ent_id]
        return EntityPair(doc_id, res_sent, f_ent, t_ent)

    def __call__(self, sents):
        samples = []
        for sent in sents:
            kk = self.extract_candidate_rels(sent.ents)
            for f_ent, t_ent in self.extract_candidate_rels(sent.ents):
                entity_pair = self.make_entity_pair(sent, f_ent, t_ent)
                samples.append(entity_pair)
        return samples


if __name__ == '__main__':
    from utils.reader import read_annotation
    path = 'classes_data'  # classes_data has deleted
    ann_data_dir = 'annotation.conf'
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