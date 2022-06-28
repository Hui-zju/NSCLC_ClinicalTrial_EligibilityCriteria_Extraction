import os
import spacy
import numpy as np
from numpy import *
from transformers import BertTokenizer
from utils.Classes import Documents, SentenceExtractor, EntityPairsExtractor
from utils.reader import read_annotation

path = '../data/variation_re_dataset/variation_re_training_dataset'  # classes_data has deleted
ann_data_dir = 'annotation.conf'
_, rel_set = read_annotation(ann_data_dir)
rel_name_list = list(set(map(lambda x: x[2], rel_set)))
rel_name_list.sort()
rel2idx = dict(zip(rel_name_list, range(1, len(rel_name_list) + 1)))

docs = Documents(path)
sent_extractor = SentenceExtractor(sent_split_char='\n', window_size=1, rel_types=rel_set, filter_no_rel_candidates_sents=False)  #

sents = sent_extractor(docs)

count_dict = {}
for sent in sents:
    for ent in sent.ents.ents:
        if ent.category not in count_dict:
            count_dict[ent.category] = 1
        else:
            count_dict[ent.category] += 1
        if ent.category == 'variation_type':
            print(ent.text)
    for rel in sent.rels.rels:
        if rel.category not in count_dict:
            count_dict[rel.category] = 1
        else:
            count_dict[rel.category] += 1
print(count_dict)
print(len({sent.doc_id[:-4] for sent in sents}))
variation_sents = [sent.text for sent in sents]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
nlp = spacy.load('en_core_sci_sm')
sentence_char_num = []
sentence_word_num = []
sentence_tokenizer_num = []
for sentence in variation_sents:
    doc = nlp(sentence)
    word_num = len(doc)
    char_num = len(sentence)
    tokenizer_num = len(tokenizer.encode(sentence))
    sentence_word_num.append(word_num)
    sentence_char_num.append(char_num)
    sentence_tokenizer_num.append(tokenizer_num)
print('sentence_word_num')
print(max(sentence_word_num))
print(mean(sentence_word_num))
print(np.percentile(sentence_word_num,[50,75,98]))
print('sentence_char_num')
print(max(sentence_char_num))
print(mean(sentence_char_num))
print(np.percentile(sentence_char_num,[50,75,98]))
print('sentence_tokenizer_num')
print(max(sentence_tokenizer_num))
print(mean(sentence_tokenizer_num))
print(np.percentile(sentence_tokenizer_num,[50,75,98]))





def sentence_length():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    nlp = spacy.load('en_core_sci_sm')

    file_path = 'trial'
    files = os.listdir(file_path)
    sentence_char_num = []
    sentence_word_num = []
    sentence_tokenizer_num = []
    for i, file in enumerate(files):
        print(file)
        if file.endswith('.txt'):
            with open('trial/' + file, 'r') as f:
                sentences = f.readlines()
                for sentence in sentences:
                    doc = nlp(sentence)
                    word_num = len(doc)
                    char_num = len(sentence)
                    tokenizer_num = len(tokenizer.encode(sentence))
                    sentence_word_num.append(word_num)
                    sentence_char_num.append(char_num)
                    sentence_tokenizer_num.append(tokenizer_num)
        if i > 500:
            break

    print('sentence_word_num')
    print(max(sentence_word_num))
    print(mean(sentence_word_num))
    print(np.percentile(sentence_word_num,[50,75,98]))
    print('sentence_char_num')
    print(max(sentence_char_num))
    print(mean(sentence_char_num))
    print(np.percentile(sentence_char_num,[50,75,98]))
    print('sentence_tokenizer_num')
    print(max(sentence_tokenizer_num))
    print(mean(sentence_tokenizer_num))
    print(np.percentile(sentence_tokenizer_num,[50,75,98]))