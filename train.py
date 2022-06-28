import os
import json
from tqdm import tqdm
import itertools
import wandb
import config
import numpy as np
import copy
import torch
import glob
import time
import logging
import torch.nn as nn
import torch.optim as optim
from glove import Glove
from pprint import pprint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, BertTokenizerFast
from common.utils import Preprocessor, DefaultLogger
from utils.reader import read_annotation
from tplinker import (HandshakingTaggingScheme,
                      DataMaker4Bert,
                      DataMaker4BiLSTM,
                      TPLinkerBert,
                      TPLinkerBiLSTM,
                      MetricsCalculator)
config = config.train_config
hyper_parameters = config["hyper_parameters"]
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# for reproductivity
torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
torch.backends.cudnn.deterministic = True


data_home = config["data_home"]
experiment_name = config["exp_name"]
train_data_path = os.path.join(data_home, experiment_name, config["train_data"])
valid_data_path = os.path.join(data_home, experiment_name, config["valid_data"])
ann_data_path = os.path.join(data_home, experiment_name, config["ann_data_dir"])


if config["logger"] == "wandb":
    # init wandb
    wandb.init(project = experiment_name,
               name = config["run_name"],
               config = hyper_parameters # Initialize config
              )

    wandb.config.note = config["note"]

    model_state_dict_dir = wandb.run.dir
    logger = wandb
else:
    logger = DefaultLogger(config["log_path"], experiment_name, config["run_name"], config["run_id"], hyper_parameters)
    model_state_dict_dir = config["path_to_save_model"]
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)


train_data = json.load(open(train_data_path, "r", encoding = "utf-8"))
valid_data = json.load(open(valid_data_path, "r", encoding = "utf-8"))

if config["encoder"] == "BERT":
    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens = False, do_lower_case = False)
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping = True, add_special_tokens = False)["offset_mapping"]
else:
    raise ValueError('config["encoder"] error')

preprocessor = Preprocessor(tokenize_func=tokenize, get_tok2char_span_map_func=get_tok2char_span_map)

# train and valid max token num
max_tok_num = 0
all_data = train_data + valid_data

for sample in all_data:
    tokens = tokenize(sample["text"])
    max_tok_num = max(max_tok_num, len(tokens))

if max_tok_num > hyper_parameters["max_seq_len"]:
    train_data = preprocessor.split_into_short_samples(train_data,
                                                          hyper_parameters["max_seq_len"],
                                                          sliding_len = hyper_parameters["sliding_len"],
                                                          encoder = config["encoder"]
                                                         )
    valid_data = preprocessor.split_into_short_samples(valid_data,
                                                          hyper_parameters["max_seq_len"],
                                                          sliding_len = hyper_parameters["sliding_len"],
                                                          encoder = config["encoder"]
                                                         )

max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])

ent_set, rel_set = read_annotation(ann_data_path)
ent_name_list = list(ent_set)
ent_name_list.sort()
token_name_list = []
for ent_name in ent_name_list:
    if ent_name.startswith('sent-'):
        token_name_list.append(ent_name)
    else:
        token_name_list.append('B-' + ent_name)
        token_name_list.append('I-' + ent_name)
tag2idx_token = dict(zip(token_name_list, range(1, len(token_name_list) + 1)))

rel_name_list = list(set(map(lambda x: x[2], rel_set)))
rel_name_list.sort()
rel_name_list += [rel_name + '_reversed' for rel_name in rel_name_list]
tag2idx_rel = dict(zip(rel_name_list, range(1, len(rel_name_list) + 1)))


handshaking_tagger = HandshakingTaggingScheme(tok2id=tag2idx_token, rel2id=tag2idx_rel, max_seq_len=max_seq_len)

if config["encoder"] == "BERT":
    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens = False, do_lower_case = False)
    data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)
else:
    raise ValueError('config["encoder"] error')


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


indexed_train_data = data_maker.get_indexed_data(train_data, max_seq_len)
indexed_valid_data = data_maker.get_indexed_data(valid_data, max_seq_len)

train_dataloader = DataLoader(MyDataset(indexed_train_data),
                                  batch_size = hyper_parameters["batch_size"],
                                  shuffle = True,
                                  num_workers = 0,
                                  drop_last = False,
                                  collate_fn = data_maker.generate_batch,
                                 )
valid_dataloader = DataLoader(MyDataset(indexed_valid_data),
                          batch_size=hyper_parameters["batch_size"],
                          shuffle=True,
                          num_workers=0,
                          drop_last = False,
                          collate_fn = data_maker.generate_batch,
                         )

if config["encoder"] == "BERT":
    encoder = AutoModel.from_pretrained(config["bert_path"])
    hidden_size = encoder.config.hidden_size
    fake_inputs = torch.zeros([hyper_parameters["batch_size"], max_seq_len, hidden_size]).to(device)
    rel_extractor = TPLinkerBert(encoder,
                                 len(tag2idx_token) + 1,
                                 len(tag2idx_rel) + 1,
                                 hyper_parameters["shaking_type"],
                                 hyper_parameters["inner_enc_type"],
                                 hyper_parameters["dist_emb_size"],
                                 hyper_parameters["ent_add_dist"],
                                 hyper_parameters["rel_add_dist"],
                                )
else:
    raise ValueError('config["encoder"] error')

rel_extractor = rel_extractor.to(device)


def bias_loss(weights=None):
    if weights is not None:
        weights = torch.FloatTensor(weights).to(device)
    cross_en = nn.CrossEntropyLoss(weight = weights)
    return lambda pred, target: cross_en(pred.view(-1, pred.size()[-1]), target.view(-1))


loss_func = bias_loss()
metrics = MetricsCalculator(handshaking_tagger)


# train step
def train_step(batch_train_data, optimizer, loss_weights):
    if config["encoder"] == "BERT":
        sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_train_data

        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_shaking_tag, \
        batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
                                                                batch_input_ids.to(device),
                                                                batch_attention_mask.to(device),
                                                                batch_token_type_ids.to(device),
                                                                batch_ent_shaking_tag.to(device),
                                                                batch_head_rel_shaking_tag.to(device),
                                                                batch_tail_rel_shaking_tag.to(device)
        )

    else:
        raise ValueError('config["encoder"] error')

    # zero the parameter gradients
    optimizer.zero_grad()

    if config["encoder"] == "BERT":
        rel_shaking_outputs, first_ent_shaking_outputs, second_ent_shaking_outputs = rel_extractor(batch_input_ids,
                                                                                                batch_attention_mask,
                                                                                                batch_token_type_ids,
                                                                                                )
    else:
        raise ValueError('config["encoder"] error')

    w_ent, w_rel = loss_weights["ent"], loss_weights["rel"]
    loss = w_ent * loss_func(first_ent_shaking_outputs, batch_ent_shaking_tag) + \
           w_ent * loss_func(second_ent_shaking_outputs, batch_head_rel_shaking_tag) + \
           w_rel * loss_func(rel_shaking_outputs, batch_tail_rel_shaking_tag)

    loss.backward()
    optimizer.step()

    rel_sample_acc = metrics.get_sample_accuracy(rel_shaking_outputs, batch_ent_shaking_tag)
    first_ent_sample_acc = metrics.get_sample_accuracy(first_ent_shaking_outputs, batch_head_rel_shaking_tag)
    second_ent_sample_acc = metrics.get_sample_accuracy(second_ent_shaking_outputs, batch_tail_rel_shaking_tag)
    # rel_cpg = metrics.get_rel_cpg(sample_list, tok2char_span_list,
    #                               rel_shaking_outputs,
    #                               first_ent_shaking_outputs,
    #                               second_ent_shaking_outputs,
    #                               hyper_parameters["match_pattern"]
    #                               )
    return loss.item(), rel_sample_acc.item(), first_ent_sample_acc.item(), second_ent_sample_acc.item()


# valid step
def valid_step(batch_valid_data):
    if config["encoder"] == "BERT":
        sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data

        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_shaking_tag, \
        batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
                                                                batch_input_ids.to(device),
                                                                batch_attention_mask.to(device),
                                                                batch_token_type_ids.to(device),
                                                                batch_ent_shaking_tag.to(device),
                                                                batch_head_rel_shaking_tag.to(device),
                                                                batch_tail_rel_shaking_tag.to(device)
        )

    else:
        raise ValueError('config["encoder"] error')

    with torch.no_grad():
        if config["encoder"] == "BERT":
            rel_shaking_outputs, first_ent_shaking_outputs, second_ent_shaking_outputs = rel_extractor(batch_input_ids,
                                                                                                    batch_attention_mask,
                                                                                                    batch_token_type_ids,
                                                                                                    )
        else:
            raise ValueError('config["encoder"] error')

    ent_sample_acc = metrics.get_sample_accuracy(rel_shaking_outputs,
                                                 batch_ent_shaking_tag)
    head_rel_sample_acc = metrics.get_sample_accuracy(first_ent_shaking_outputs,
                                                      batch_head_rel_shaking_tag)
    tail_rel_sample_acc = metrics.get_sample_accuracy(second_ent_shaking_outputs,
                                                      batch_tail_rel_shaking_tag)

    rel_cpg = metrics.get_rel_cpg(sample_list, tok2char_span_list,
                                  rel_shaking_outputs,
                                  first_ent_shaking_outputs,
                                  second_ent_shaking_outputs,
                                  hyper_parameters["match_pattern"]
                                  )

    return ent_sample_acc.item(), head_rel_sample_acc.item(), tail_rel_sample_acc.item(), rel_cpg


max_f1 = 0.


def train_n_valid(train_dataloader, dev_dataloader, optimizer, scheduler, num_epoch):
    def train(dataloader, ep):
        # train
        rel_extractor.train()

        t_ep = time.time()
        start_lr = optimizer.param_groups[0]['lr']
        total_loss, total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0., 0.
        for batch_ind, batch_train_data in enumerate(dataloader):
            t_batch = time.time()
            z = (2 * len(tag2idx_rel) + 1)
            steps_per_ep = len(dataloader)
            total_steps = hyper_parameters["loss_weight_recover_steps"] + 1  # + 1 avoid division by zero error
            current_step = steps_per_ep * ep + batch_ind
            w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
            w_rel = min((len(tag2idx_rel) / z) * current_step / total_steps, (len(tag2idx_rel) / z))
            loss_weights = {"ent": w_ent, "rel": w_rel}

            loss, ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc = train_step(batch_train_data, optimizer,
                                                                                        loss_weights)
            scheduler.step()

            total_loss += loss
            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc

            avg_loss = total_loss / (batch_ind + 1)
            avg_rel_sample_acc = total_ent_sample_acc / (batch_ind + 1)
            avg_first_ent_sample_acc = total_head_rel_sample_acc / (batch_ind + 1)
            avg_second_ent_sample_acc = total_tail_rel_sample_acc / (batch_ind + 1)

            batch_print_format = "\rEpoch: {}/{}, batch: {}/{}, train_loss: {:.6f}, " + "t_rel_sample_acc: {:.4f}, t_first_ent_sample_acc: {:.4f}, t_second_ent_sample_acc: {:.4f}, lr: {:.6f},"
            # + "project: {}, run_name: {}, batch_time: {}, total_time: {}-------------"
            print(batch_print_format.format(
                                            ep + 1, num_epoch,
                                            batch_ind + 1, len(dataloader),
                                            avg_loss,
                                            avg_rel_sample_acc,
                                            avg_first_ent_sample_acc,
                                            avg_second_ent_sample_acc,
                                            optimizer.param_groups[0]['lr'],
                                            ), end="")
            # experiment_name,
            # config["run_name"],
            # time.time() - t_batch,
            # time.time() - t_ep,

            if config["logger"] == "wandb" and batch_ind % hyper_parameters["log_interval"] == 0:
                logger.log({
                    "train_loss": avg_loss,
                    "train_rel_acc": avg_rel_sample_acc,
                    "train_first_ent_acc": avg_first_ent_sample_acc,
                    "train_second_ent_acc": avg_second_ent_sample_acc,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "time": time.time() - t_ep,
                })

        if config["logger"] != "wandb":  # only log once for training if logger is not wandb
            print('\n')
            logger.log({
                "train_loss": avg_loss,
                "train_rel_acc": avg_rel_sample_acc,
                "train_first_ent_acc": avg_first_ent_sample_acc,
                "train_second_ent_acc": avg_second_ent_sample_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "time": time.time() - t_ep,
            })

    def valid(dataloader, ep):
        # valid
        rel_extractor.eval()

        t_ep = time.time()
        total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0.
        total_rel_correct_num, total_rel_pred_num, total_rel_gold_num = 0, 0, 0
        for batch_ind, batch_valid_data in enumerate(tqdm(dataloader, desc="Validating")):
            ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc, rel_cpg = valid_step(batch_valid_data)

            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc

            total_rel_correct_num += rel_cpg[0]
            total_rel_pred_num += rel_cpg[1]
            total_rel_gold_num += rel_cpg[2]

        avg_rel_sample_acc = total_ent_sample_acc / len(dataloader)
        avg_first_ent_sample_acc = total_head_rel_sample_acc / len(dataloader)
        avg_second_ent_sample_acc = total_tail_rel_sample_acc / len(dataloader)

        rel_prf = metrics.get_prf_scores(total_rel_correct_num, total_rel_pred_num, total_rel_gold_num)

        log_dict = {
            "val_rel_acc": avg_rel_sample_acc,
            "val_first_ent_acc": avg_first_ent_sample_acc,
            "val_second_ent_acc": avg_second_ent_sample_acc,
            "val_prec": rel_prf[0],
            "val_recall": rel_prf[1],
            "val_f1": rel_prf[2],
            "time": time.time() - t_ep,
        }
        logger.log(log_dict)
        pprint(log_dict)

        return rel_prf[2]

    for ep in range(num_epoch):
        train(train_dataloader, ep)
        valid_f1 = valid(valid_dataloader, ep)

        global max_f1
        if valid_f1 >= max_f1:
            max_f1 = valid_f1
            if valid_f1 > config["f1_2_save"]:  # save the best model
                modle_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                torch.save(rel_extractor.state_dict(),
                           os.path.join(model_state_dict_dir, "model_state_dict_{}.pt".format(modle_state_num)))
        #                 scheduler_state_num = len(glob.glob(schedule_state_dict_dir + "/scheduler_state_dict_*.pt"))
        #                 torch.save(scheduler.state_dict(), os.path.join(schedule_state_dict_dir, "scheduler_state_dict_{}.pt".format(scheduler_state_num)))
        print("Current avf_f1: {}, Best f1: {}".format(valid_f1, max_f1))


init_learning_rate = float(hyper_parameters["lr"])
optimizer = torch.optim.Adam(rel_extractor.parameters(), lr=init_learning_rate)

if hyper_parameters["scheduler"] == "CAWR":
    T_mult = hyper_parameters["T_mult"]
    rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     len(train_dataloader) * rewarm_epoch_num, T_mult)
elif hyper_parameters["scheduler"] == "Step":
    decay_rate = hyper_parameters["decay_rate"]
    decay_steps = hyper_parameters["decay_steps"]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
else:
    raise ValueError('hyper_parameters["scheduler"] error')

if not config["fr_scratch"]:
    model_state_path = config["model_state_dict_path"]
    rel_extractor.load_state_dict(torch.load(model_state_path))
    print("------------model state {} loaded ----------------".format(model_state_path.split("/")[-1]))

train_n_valid(train_dataloader, valid_dataloader, optimizer, scheduler, hyper_parameters["epochs"])