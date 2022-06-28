import re
import os
import unicodedata
import spacy
import json
nlp = spacy.load('en_core_sci_sm')

# 提取入排标准
# 输入EligibilityCriteria字符串，返回清洗后EligibilityCriteria字符串
def clean_data(eligibility_criteria):
    eligibility_criteria = re.sub('[\n]+', '\n', eligibility_criteria)  # 去除多余的空行
    eligibility_criteria = re.sub('  ', ' ', eligibility_criteria)  # 去除多余的空格
    eligibility_criteria = unicodedata.normalize('NFKC', eligibility_criteria)  # 将中文标点转为英文标点
    eligibility_criteria_rows = eligibility_criteria.split('\n')  # 分为不同的行
    # 对于没有标点的行，在末尾添加.（便于再分句）
    for index, sentence in enumerate(eligibility_criteria_rows):
        if eligibility_criteria_rows[index][-1] not in [',', '.', ';', ':', '?']:
            eligibility_criteria_rows[index] = eligibility_criteria_rows[index] + '. '

    # 将过长的行再分句
    eligibility_criteria_sentences = []
    for row in eligibility_criteria_rows:
        doc = nlp(row)
        for sent in doc.sents:
            sent = str(sent)
            eligibility_criteria_sentences.append(sent)

    # split by ';'
    eligibility_criteria_sentences = [sent for sentence in eligibility_criteria_sentences for sent in sentence.split(';')]

    # 去除字符数量不超过3个的句子（无意义的句子）
    old_eligibility_criteria_sentences = eligibility_criteria_sentences
    eligibility_criteria_sentences = []
    for index in range(0, len(old_eligibility_criteria_sentences)):
        if len(old_eligibility_criteria_sentences[index]) > 3:
            eligibility_criteria_sentences.append(old_eligibility_criteria_sentences[index])
    eligibility_criteria = '\n'.join(eligibility_criteria_sentences)
    return eligibility_criteria


# 输入EligibilityCriteria字符串，返回inclusion和exclusion字符串
def split_eligibility_criteria(eligibility_criteria):
    eligibility_criteria_sentences = eligibility_criteria.split('\n')  # 分句
    inclusion_sentences = []
    exclusion_sentences = []
    exclusion_flag = 1
    for sentence in eligibility_criteria_sentences:
        if re.search(r'exclusion|exlusion', sentence, flags=re.I) != None and len(sentence) < 50:
            exclusion_flag = 0
        if exclusion_flag and len(sentence) > 0:
            inclusion_sentences.append(sentence)
        else:
            exclusion_sentences.append(sentence)
    inclusion_txt = '\n'.join(inclusion_sentences)
    exclusion_txt = '\n'.join(exclusion_sentences)
    return inclusion_txt, exclusion_txt


def build_eligibility_criteria_on_file(json_data, target_folder_path):
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)
    data = json_data["FullStudiesResponse"]["FullStudies"][0]["Study"]
    NCTId = data["ProtocolSection"]["IdentificationModule"]["NCTId"]
    EligibilityCriteria = data["ProtocolSection"]["EligibilityModule"]["EligibilityCriteria"]
    eligibility_criteria = clean_data(EligibilityCriteria)
    inclusion_txt, exclusion_txt = split_eligibility_criteria(eligibility_criteria)
    print(NCTId)
    with open(os.path.join(target_folder_path, NCTId + '_inc.txt'), 'w') as f:
        f.write(inclusion_txt)
    with open(os.path.join(target_folder_path, NCTId + '_exc.txt'), 'w') as f:
        f.write(exclusion_txt)


def build_eligibility_criteria_on_folder(json_data_folder_path, target_folder_path):
    for json_file in os.listdir(json_data_folder_path):
        if json_file.endswith('.json'):
            json_file_path = os.path.join(json_data_folder_path, json_file)
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
            build_eligibility_criteria_on_file(json_data, target_folder_path)


def build_brief_summary_on_file(json_data, target_folder_path):
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)
    data = json_data["FullStudiesResponse"]["FullStudies"][0]["Study"]
    NCTId = data["ProtocolSection"]["IdentificationModule"]["NCTId"]
    brief_summary = data["ProtocolSection"]["DescriptionModule"]["BriefSummary"]
    brief_summary = clean_data(brief_summary)
    print(NCTId)
    with open(os.path.join(target_folder_path, NCTId + '_sum.txt'), 'w') as f:
        f.write(brief_summary)


def build_extraction_dataset_on_folder(json_data_folder_path, target_folder_path):
    for json_file in os.listdir(json_data_folder_path):
        if json_file.endswith('.json'):
            json_file_path = os.path.join(json_data_folder_path, json_file)
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
            build_eligibility_criteria_on_file(json_data, target_folder_path)
            build_brief_summary_on_file(json_data, target_folder_path)

