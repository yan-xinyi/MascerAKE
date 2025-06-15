#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化的BERT模型脚本，用于HPC Stanage服务器上运行
"""

import transformers
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertTokenizerFast
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.optim import AdamW
from nltk.corpus import wordnet
import math
import os
import logging
import argparse
from datetime import datetime
from gensim.models import FastText
from sklearn.cluster import AgglomerativeClustering
import nltk
nltk.download('wordnet')
# 设置使用GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 设置日志格式
def setup_logger(log_path):
    """设置日志配置"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# 常量定义
FLOAT_TOLERANCE = 1e-6
OOV_FEATURE_TEMPLATE = [
    0.19849130316149868, 0.6495057034220533, 0.172319391634981, 0.18604562737642585, 0.9076235741444868,
    0.19098859315589356, 0.5134030418250951, 0.24509505703422055, 0.032338403041825094, 0.8642015209125475,
    0.8871102661596958, 0.9170532319391634, 0.9219391634980989, 0.8235171102661597, 0.62606463878327,
    0.6616539923954373, 0.6707224334600761, 0.8414258555133081, 7.604562737642586e-05, 0.0,
    0.8453802281368821, 0.9555133079847908, 0.9981368821292775, 1.0, 0.9194296577946768
]
OOV_FEATURE = [0.01] * 25

# Load BNC Corpus for Frequency Normalization
bnc_path = '/users/li4xy/AKE/datas/lemma.al'
word_freq = {}
with open(bnc_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            word, freq = parts[2], int(parts[1])
            word_freq[word] = freq

# 全局变量，将在main函数内进行初始化
wordid_to_word = {}
word_to_wordid = {}
train_oov_dict, train_cognitive_dict = {}, {}
test_oov_dict, test_cognitive_dict = {}, {}
word_index_counter = 0  # 用于分配词id
vocab_feature_dict = {}
oov_feature_dict = {}
fasttext_dict = {}


def is_same_as_oov_template(features):
    return all(
        math.isclose(float(f), t, abs_tol=FLOAT_TOLERANCE)
        for f, t in zip(features, OOV_FEATURE_TEMPLATE)
    )

def compute_log_freq(word, word_freq):
    """获取词频的对数加权值"""
    freq = word_freq.get(word, 1)
    return math.log10(freq * 10)

def regularize_features_by_word_length(template, word):
    """使用词长调整OOV模板特征"""
    word_len = len(word)
    return [val * word_len for val in template]


def get_features(item, word, OOV=str, mode=str):
    """
    获取词的认知特征，处理OOV情况

    Args:
        item: 当前词条数据
        word: 词本身
        mode: 'train'或'test'模式

    Returns:
        features: 提取的特征向量
    """
    global word_index_counter

    word_lower = word.lower()
    # 提取原始特征（17维眼动 + 8维脑电）
    et_features = [float(item[n + 1]) for n in range(17)]
    eeg_features = [float(item[m + 18]) for m in range(8)]
    features = et_features + eeg_features

    # 如果为 OOV 特征，处理并记录
    if is_same_as_oov_template(features):
        if mode == "train":
            if OOV == "zero":
                train_oov_dict[word_lower] = OOV_FEATURE        # 消融"YAN均值方法"
            if OOV == "yan":
                train_oov_dict[word_lower] = OOV_FEATURE_TEMPLATE  # "YAN均值方法"
            if OOV == "zhang":
                train_oov_dict[word_lower] = OOV_FEATURE
                train_oov_dict[word_lower] = regularize_features_by_word_length(OOV_FEATURE, word_lower)  # "Zhang方法"
            if OOV == "yan+zhang":
                train_oov_dict[word_lower] = OOV_FEATURE_TEMPLATE
                train_oov_dict[word_lower] = regularize_features_by_word_length(OOV_FEATURE_TEMPLATE, word_lower)   # "Zhang+YAN方法"

        else:
            if OOV == "zero":
                test_oov_dict[word_lower] = OOV_FEATURE
            if OOV == "yan":
                test_oov_dict[word_lower] = OOV_FEATURE_TEMPLATE
            if OOV == "zhang":
                train_oov_dict[word_lower] = OOV_FEATURE
                test_oov_dict[word_lower] = regularize_features_by_word_length(OOV_FEATURE, word_lower)  # "Zhang方法"
            if OOV == "yan+zhang":
                test_oov_dict[word_lower] = OOV_FEATURE_TEMPLATE
                test_oov_dict[word_lower] = regularize_features_by_word_length(OOV_FEATURE_TEMPLATE, word_lower)   # "Zhang+YAN方法"
    else:
        if mode == "train":
            train_cognitive_dict[word_lower] = features
        else:
            test_cognitive_dict[word_lower]  = features

    # 更新词ID映射
    if word_lower not in word_to_wordid:
        word_to_wordid[word_lower] = word_index_counter
        wordid_to_word[word_index_counter] = word_lower
        word_index_counter += 1

    return features


def merge_cognitive_dicts(train_dict, test_dict):
    """合并训练和测试数据字典，处理冲突"""
    merged_dict = dict(train_dict)  # 先复制 train 的内容
    for key, value in test_dict.items():
        if key in merged_dict:
            if merged_dict[key] != value:
                logging.warning(f"Conflict for key '{key}': values differ between train and test.")
        else:
            merged_dict[key] = value
    return merged_dict


def extract_sentences_from_dataloader(loader, tokenizer):
    """从DataLoader中提取句子，用于FastText训练"""
    sentences = []
    for batch in loader:
        # 检查 batch 的数据结构
        if isinstance(batch, (tuple, list)):
            input_ids = batch[0]  # 假设 input_ids 是第一个元素
        elif isinstance(batch, dict):
            input_ids = batch['input_ids']
        else:
            raise ValueError("Unexpected batch format.")

        # 将 input_ids 转换为 tokens
        for input_id in input_ids:
            tokens = tokenizer.convert_ids_to_tokens(input_id)
            sentence = tokenizer.convert_tokens_to_string(tokens)
            sentences.append(sentence)
    return sentences


class MyDataset(Dataset):
    """自定义数据集类，处理文本和特征对齐"""

    def __init__(self, texts, old_features, tags):
        self.texts = texts
        self.tags = tags
        self.old_features = old_features

        self.labels = []
        self.tokens = []
        self.features = []

        self.input_ids = None
        self.attention_masks = None

    def encode(self, tokenizer, max_len, label_to_ids, label_all_tokens=True):
        """编码文本和标签，对齐特征"""
        for i in tqdm(range(len(self.texts)), desc="Encoding dataset"):
            text = self.texts[i]
            tag = self.tags[i]
            feature = self.old_features[i]
            tags, tokens, features = self._align_label(text, tag, feature,
                                                       tokenizer, max_len,
                                                       label_to_ids, label_all_tokens)
            self.labels.append(tags)
            self.tokens.append(tokens)
            self.features.append(features)

        self.features = [torch.tensor(f, dtype=torch.float32) for f in self.features]

        self.inputs = tokenizer(self.texts, max_length=max_len, add_special_tokens=True,
                                padding='max_length', truncation=True, return_tensors='pt')
        self.input_ids = self.inputs['input_ids']
        self.attention_masks = self.inputs['attention_mask']

    def _align_label(self, text, labels, features, tokenizer, max_len, label_to_ids, label_all_tokens):
        """对齐文本标签和特征"""
        input = tokenizer(text, max_length=max_len, add_special_tokens=True,
                          padding='max_length', truncation=True, return_tensors='pt')
        word_ids = input.word_ids()
        input_ids = input['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        previous_word_idx = None
        new_labels = []
        new_features = []
        no_features = [0 for i in range(25)]  # 空特征向量

        for word_idx in word_ids:
            if word_idx is None:
                new_labels.append('none')
                new_features.append(no_features)
            elif word_idx != previous_word_idx:
                try:
                    new_labels.append(labels[word_idx])
                    new_features.append(features[word_idx])
                except:
                    new_labels.append('none')
                    new_features.append(no_features)
            else:
                try:
                    new_labels.append(labels[word_idx] if label_all_tokens else 'none')
                    new_features.append(features[word_idx] if label_all_tokens else no_features)
                except:
                    new_labels.append('none')
                    new_features.append(no_features)
            previous_word_idx = word_idx

        label_ids = [label_to_ids[label] for label in new_labels]

        return label_ids, tokens, new_features

    def __getitem__(self, idx):
        """返回单个样本"""
        return (self.input_ids[idx, :],
                self.attention_masks[idx, :],
                self.tokens[idx],
                torch.tensor(self.features[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx]))

    def __len__(self):
        """返回数据集大小"""
        return len(self.input_ids)

def leacock_chodorow_similarity(c1, c2, D=16):
    """
    Leacock-Chodorow 相似度，用于两个 synsets。
    D 是 WordNet 的最大深度，经验值为 16（可调）。
    """
    try:
        sp = c1.shortest_path_distance(c2)
        if sp is None or sp == 0:
            return 0.0
        return -math.log(sp / (2.0 * D))
    except:
        return 0.0

def compute_distance_matrix(synsets, D=16):
    """
    构造 synsets 的 pairwise 距离矩阵，使用 Leacock-Chodorow 相似度。
    距离 = 1 - (LCH 相似度 / 最大相似度)
    """
    n = len(synsets)
    distance_matrix = np.zeros((n, n))
    max_lch = -math.log(1 / (2.0 * D))  # 相似度的最大值（最短路径为 1）

    for i in range(n):
        for j in range(n):
            if i == j:
                distance = 0.0
            else:
                sim = leacock_chodorow_similarity(synsets[i], synsets[j], D)
                # 将相似度转换为距离（归一化到 [0, 1]）
                distance = 1.0 - (sim / max_lch) if sim > 0 else 1.0
            distance_matrix[i][j] = distance

    return distance_matrix

def get_synsets(words):
    synsets = []
    for word in words:
        synsets.extend(wordnet.synsets(word))
    return synsets

def cluster_synsets_by_similarity(words, sim_threshold=0.2):
    """
    对 WordNet 同义词集聚类，返回 cluster_id -> [synsets]
    """
    synsets = []
    for word in words:
        synsets.extend(wordnet.synsets(word))

    if len(synsets) < 2:
        if not synsets:
            return {}
        else:
            return {syn.name(): [syn] for syn in synsets}

    distance_matrix = compute_distance_matrix(synsets)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        distance_threshold=3.5 - sim_threshold,  # 可调节
        linkage='single'
    )
    labels = clustering.fit_predict(distance_matrix)

    cluster_map = {}
    for label, synset in zip(labels, synsets):
        cluster_map.setdefault(label, []).append(synset)

    return cluster_map


def average_cluster_features(cluster_map, vocab_feature_dict, device, feature_dim):
    cluster_feature_map = {}

    for cluster_id, synsets in cluster_map.items():
        vectors = []
        for synset in synsets:
            for lemma in synset.lemmas():
                name = lemma.name().lower()
                if name in vocab_feature_dict:
                    vectors.append(torch.tensor(vocab_feature_dict[name], dtype=torch.float))
                    break
        if vectors:
            cluster_vector = torch.mean(torch.stack(vectors), dim=0).to(device)
            for synset in synsets:
                cluster_feature_map[synset.name()] = cluster_vector

    return cluster_feature_map


class BertNerModel(nn.Module):
    """集成认知特征的BERT命名实体识别模型"""

    def __init__(self, bert_weight, num_labels=6, fs_num=0, tokenizer=None, vocab_feature_dict={}, oov_feature_dict={},
                 fasttext_dict={}, fs_combine="base"):
        super(BertNerModel, self).__init__()

        self.bert = BertModel.from_pretrained(bert_weight)
        self.tokenizer = tokenizer
        self.vocab_feature_dict = vocab_feature_dict
        self.oov_feature_dict = oov_feature_dict
        self.fasttext_dict = fasttext_dict

        self.fs_combine = fs_combine
        self.hidden_size = 768
        if fs_combine == "et":
            self.hidden_size += 17
        elif fs_combine == "eeg":
            self.hidden_size += 8
        elif fs_combine == "beta+gamma":
            self.hidden_size += 4
        elif fs_combine == "beta" or "gamma":
            self.hidden_size += 2
        elif fs_combine == "et+eeg":
            self.hidden_size += 25

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def get_synonym_embedding(self, token):
        """查找token的同义词并返回存在于词汇表中的同义词嵌入"""
        synonyms = wordnet.synsets(token)
        for syn in synonyms:
            for lemma in syn.lemmas():
                synonym = lemma.name().lower()
                if synonym in self.vocab_feature_dict:
                    wornet_value = self.vocab_feature_dict[synonym]
                    return wornet_value
        return None

    def replace_oov_embeddings_and_features(self, input_ids, pooled_outputs, extra_features,
                                            wordnet_weight=0.6, fasttext_weight=0.4):
        """
        替换 OOV 词的特征表示，融合 WordNet 同义词嵌入和 FastText 向量
        """
        with torch.no_grad():
            batch_size, seq_len = input_ids.size()
            for i in range(batch_size):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                word_level_map = []                     # 映射 token 到原始词
                current_word = ''
                for tok in tokens:
                    if tok.startswith("##"):
                        current_word += tok[2:]
                    else:
                        current_word = tok
                    word_level_map.append(current_word.lower())

                for j, word in enumerate(word_level_map):
                    if word in self.oov_feature_dict:           # 是 OOV 词
                        synonym_feature = self.get_synonym_embedding(word)
                        fasttext_feature = self.fasttext_dict.get(word, None)

                        if synonym_feature is not None and fasttext_feature is not None:
                            combined_feat = [
                                wordnet_weight * s + fasttext_weight * f
                                for s, f in zip(synonym_feature, fasttext_feature)
                            ]
                            extra_features[i, j, :] = torch.tensor(combined_feat, dtype=extra_features.dtype,
                                                                   device=extra_features.device)
                        elif synonym_feature is not None:
                            extra_features[i, j, :] = torch.tensor(synonym_feature, dtype=extra_features.dtype,
                                                                   device=extra_features.device)
                        elif fasttext_feature is not None:
                            extra_features[i, j, :] = torch.tensor(fasttext_feature, dtype=extra_features.dtype,
                                                                   device=extra_features.device)


    def forward(self, input_ids, attention_mask, extra_features, token_type_ids=None, wordnet_weight=0.6,
                fasttext_weight=0.4):
        """前向传播，处理BERT输出和认知特征"""
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_outputs = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 确保 extra_features 是张量，形状 [batch_size, seq_len, feature_dim]
        if isinstance(extra_features, list):
            extra_features = torch.stack(extra_features, dim=1)
        extra_features = extra_features.to(pooled_outputs.device)

        # 替换 OOV 词的认知特征
        self.replace_oov_embeddings_and_features(input_ids, pooled_outputs, extra_features,
                                                 wordnet_weight=wordnet_weight,
                                                 fasttext_weight=fasttext_weight)

        # 根据特征组合方式拼接BERT输出和认知特征
        if self.fs_combine == "et":
            combined_outputs = torch.cat((pooled_outputs, extra_features[:, :, :17]), dim=-1)
        elif self.fs_combine == "eeg":
            combined_outputs = torch.cat((pooled_outputs, extra_features[:, :, 17:25]), dim=-1)
        elif self.fs_combine == "beta":
            combined_outputs = torch.cat((pooled_outputs, extra_features[:, :, 21:23]), dim=-1)
        elif self.fs_combine == "gamma":
            combined_outputs = torch.cat((pooled_outputs, extra_features[:, :, 23:25]), dim=-1)
        elif self.fs_combine == "beta+gamma":
            combined_outputs = torch.cat((pooled_outputs, extra_features[:, :, 21:25]), dim=-1)
        elif self.fs_combine == "et+eeg":
            combined_outputs = torch.cat((pooled_outputs, extra_features), dim=-1)
        else:
            combined_outputs = pooled_outputs

        combined_outputs = self.dropout(combined_outputs)
        batch_size, seq_length, hidden_size = combined_outputs.shape
        combined_outputs = combined_outputs.view(batch_size * seq_length, hidden_size)

        logits = self.classifier(combined_outputs)
        logits = logits.view(batch_size, seq_length, -1)

        return logits


def TagConvert(raw_tags, words_set, poss=None):
    """将标签转换为关键词列表"""
    true_tags = []
    for i in range(raw_tags.shape[0]):
        kw_list = []
        nkw_list = ""
        for j in range(len(raw_tags[i])):
            item = raw_tags[i][j]
            if item == 0:
                continue
            if poss is not None and j in poss[i]:
                continue
            if item == 5:
                nkw_list = ""
            if item == 4:
                kw_list.append(str(words_set[j][i]))
            if item == 1:
                nkw_list += str(words_set[j][i])
            if item == 2:
                nkw_list += " "
                nkw_list += str(words_set[j][i])
            if item == 3:
                nkw_list += " "
                nkw_list += str(words_set[j][i])
                kw_list.append(nkw_list)
                nkw_list = ""

        true_tags.append(kw_list)
    return true_tags


def evaluate(predict_data, target_data, topk=3):
    """评估关键词提取性能"""
    TRUE_COUNT, PRED_COUNT, GOLD_COUNT = 0.0, 0.0, 0.0
    for index, words in enumerate(predict_data):
        y_pred, y_true = None, target_data[index]

        if isinstance(predict_data[index], dict):
            words = sorted(words.items(), key=lambda item: (-item[1], item[0]))
            y_pred = [i[0] for i in words]
        else:
            y_pred = words

        y_pred = y_pred[0: topk]
        TRUE_NUM = len(set(y_pred) & set(y_true))
        TRUE_COUNT += TRUE_NUM
        PRED_COUNT += len(y_pred)
        GOLD_COUNT += len(y_true)

    # 计算精确率P
    p = (TRUE_COUNT / PRED_COUNT) if PRED_COUNT != 0 else 0

    # 计算召回率R
    r = (TRUE_COUNT / GOLD_COUNT) if GOLD_COUNT != 0 else 0

    # 计算F1
    f1 = ((2 * r * p) / (r + p)) if (r + p) != 0 else 0

    p = round(p * 100, 2)
    r = round(r * 100, 2)
    f1 = round(f1 * 100, 2)

    return p, r, f1


def calculate_f1(y_pred, y_true):
    """计算F1分数，排除padding标签"""
    # 展平并转换为numpy数组
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # 排除padding标签(0)
    mask = np.where(y_true != 0)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return y_pred, y_true


def load_data(args):
    """加载训练和测试数据"""
    global train_oov_dict, train_cognitive_dict, test_oov_dict, test_cognitive_dict

    logger = logging.getLogger(__name__)
    logger.info(f"加载数据集: {args.dn}")

    # 选择数据集路径
    if args.dn == "ELE":
        train_path = args.ele_train_path
        test_path = args.ele_test_path
    elif args.dn == "GT":
        train_path = args.gt_train_path
        test_path = args.gt_test_path
    else:
        raise ValueError(f"Unknown dataset: {args.dn}")

    # 加载JSON文件
    train_file = json.load(open(train_path, 'r', encoding='utf-8'))
    test_file = json.load(open(test_path, 'r', encoding='utf-8'))

    # 处理训练数据
    train_sens, train_tags = [], []
    train_Feature = []
    train_word_nums = []

    for key1 in train_file.keys():
        tags = []
        features = []
        items = train_file[key1]
        sens = ''
        nums = 0
        for item in items:
            sens += item[0]
            word = item[0]
            sens += ' '
            fs = get_features(item, word, args.OOV, "train")
            features.append(fs)
            tags.append(item[-1])
            nums += 1
        train_sens.append(sens.strip())
        train_word_nums.append(nums)
        train_Feature.append(features)
        train_tags.append(tags)

    # 处理测试数据
    test_sens, test_tags = [], []
    test_Feature = []
    test_word_nums = []

    for key2 in test_file.keys():
        tags = []
        features = []
        items = test_file[key2]
        sens = ''
        nums = 0
        for item in items:
            sens += item[0]
            word = item[0]
            sens += ' '
            fs = get_features(item, word, args.OOV, "test")
            features.append(fs)
            tags.append(item[-1])
            nums += 1
        test_sens.append(sens.strip())
        test_word_nums.append(nums)
        test_Feature.append(features)
        test_tags.append(tags)

    logger.info(f"训练集中的非OOV词数量: {len(train_cognitive_dict)}")
    logger.info(f"训练集中的OOV词数量: {len(train_oov_dict)}")
    logger.info(f"测试集中的非OOV词数量: {len(test_cognitive_dict)}")
    logger.info(f"测试集中的OOV词数量: {len(test_oov_dict)}")

    return train_sens, train_Feature, train_tags, test_sens, test_Feature, test_tags


def prepare_fasttext(train_dataloader, test_dataloader, tokenizer, oov_feature_dict, vocab_feature_dict):
    """准备FastText模型和特征字典"""
    logger = logging.getLogger(__name__)
    logger.info("准备FastText模型...")

    # 提取用于FastText训练的句子
    train_sentences = extract_sentences_from_dataloader(train_dataloader, tokenizer)
    test_sentences = extract_sentences_from_dataloader(test_dataloader, tokenizer)
    sentences = train_sentences + test_sentences

    # 确保句子格式正确
    if isinstance(sentences, list) and all(isinstance(sentence, str) for sentence in sentences):
        sentences = [sentence.split() for sentence in sentences]
    else:
        raise ValueError("提取的句子格式不正确")

    # 训练FastText模型
    fasttext_model = FastText(
        sentences=sentences,
        vector_size=100,  # 向量维度
        window=5,
        min_count=1,
        sg=1  # Skip-gram模型
    )

    # 构建fasttext特征字典
    fasttext_dict = {}

    # 处理OOV词
    for word in list(oov_feature_dict.keys()):
        if fasttext_model is not None and word in fasttext_model.wv:
            fasttext_vec = fasttext_model.wv[word.lower()]
            features = list(fasttext_vec[:25]) if len(fasttext_vec) >= 25 else list(fasttext_vec) + [0.0] * (
                        25 - len(fasttext_vec))

            # 归一化到 0~1 之间
            min_val = min(features)
            max_val = max(features)
            if max_val > min_val:  # 避免除以零
                features = [(x - min_val) / (max_val - min_val) for x in features]
            else:
                features = [0.0] * len(features)  # 所有值都相同的情况

            fasttext_dict[word] = features
        else:
            # 如果FastText模型没覆盖，保留原来的OOV特征
            fasttext_dict[word] = oov_feature_dict[word]

    # 同时将非OOV词的特征写入fasttext_dict
    for word, features in vocab_feature_dict.items():
        if word not in fasttext_dict:
            fasttext_dict[word] = features

    logger.info(f"FastText特征字典大小: {len(fasttext_dict)}")
    logger.info(f"词汇特征字典大小: {len(vocab_feature_dict)}")
    logger.info(f"OOV特征字典大小: {len(oov_feature_dict)}")

    return fasttext_dict


def train_model(model, train_dataloader, test_dataloader, args, device, logger):
    """训练和验证模型"""
    # 设置优化器和损失函数
    optim = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=0).to(device)

    best_f1 = 0.0
    for epoch in range(args.epochs):
        # 训练阶段
        loss_value = 0.0
        model.train()
        label_true, label_pred = [], []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for i, batch in enumerate(progress_bar):
            optim.zero_grad()
            input_ids, attention_masks, _, features, tags = batch
            pred_tags = model(input_ids.to(device), attention_masks.to(device), features.to(device))

            loss = loss_fn(pred_tags.permute(0, 2, 1), tags.to(device))
            loss = loss.mean()
            loss.backward()
            optim.step()

            pred_tags = F.softmax(pred_tags, dim=-1)
            pred_tags = torch.argmax(pred_tags, dim=-1)

            y_pred, y_true = calculate_f1(pred_tags, tags)
            label_true.extend(y_true)
            label_pred.extend(y_pred)

            loss_value += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        label_train_f1 = f1_score(label_true, label_pred, average='macro')

        # 验证阶段
        model.eval()
        kw_true, kw_pred = [], []
        label_true, label_pred = [], []

        for batch in tqdm(test_dataloader, desc="Validating"):
            input_ids, attention_masks, tokens, features, tags = batch
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = 0
                        module.train(False)

                pred_tags = model(input_ids.to(device), attention_masks.to(device), features.to(device))
                pred_tags = F.softmax(pred_tags, dim=-1)
                pred_tags = torch.argmax(pred_tags, dim=-1)

            y_pred, y_true = calculate_f1(pred_tags, tags)
            label_true.extend(y_true)
            label_pred.extend(y_pred)

            # 处理关键词评估
            poss = []
            for i in range(len(tags)):
                pos = []
                for j in range(len(tags[i])):
                    if tags[i][j] == 0:
                        pos.append(j)
                poss.append(pos)

            kw_true.extend(TagConvert(tags, tokens))
            kw_pred.extend(TagConvert(pred_tags, tokens, poss))

        # 计算评估指标
        label_f1 = f1_score(label_true, label_pred, average='macro')
        P, R, F1 = evaluate(kw_pred, kw_true)

        # 保存最佳模型
        if F1 > best_f1:
            best_f1 = F1
            torch.save(model.state_dict(), args.model_path)

        # 记录日志
        log_msg = (f"Epoch {epoch + 1}: loss:{loss_value / len(train_dataloader):.2f} "
                   f"train_f1:{label_train_f1:.2f} test_f1:{label_f1:.2f} "
                   f"kw_p:{P:.2f} kw_r:{R:.2f} kw_f1:{F1:.2f}")
        logger.info(log_msg)

    return best_f1


def test_model(model, test_dataloader, device, logger):
    """测试模型并返回最终评估结果"""
    logger.info("开始最终测试...")

    model.eval()
    kw_true, kw_pred = [], []
    label_true, label_pred = [], []

    for batch in tqdm(test_dataloader, desc="Testing"):
        input_ids, attention_masks, tokens, features, tags = batch
        with torch.no_grad():
            pred_tags = model(input_ids.to(device), attention_masks.to(device), features.to(device))
            pred_tags = torch.argmax(pred_tags, dim=-1)

        y_pred, y_true = calculate_f1(pred_tags, tags)
        label_true.extend(y_true)
        label_pred.extend(y_pred)

        # 处理关键词评估
        poss = []
        for idx in range(len(tags)):
            pos = [j for j in range(len(tags[idx])) if tags[idx][j] == 0]
            poss.append(pos)

        true_keywords = TagConvert(tags, tokens)
        pred_keywords = TagConvert(pred_tags, tokens, poss)

        kw_true.extend(true_keywords)
        kw_pred.extend(pred_keywords)

    # 最终评估
    label_f1 = f1_score(label_true, label_pred, average='macro')
    P, R, F1 = evaluate(kw_pred, kw_true)

    logger.info(f"最终测试结果 - 精确率: {P}, 召回率: {R}, F1分数: {F1}")

    return P, R, F1


def main():
    """主函数，处理参数并执行完整流程"""
    global vocab_feature_dict, oov_feature_dict, fasttext_dict

    # 参数解析
    parser = argparse.ArgumentParser(description='MASOR+BERT模型训练与评估')

    # 数据集参数
    parser.add_argument('--dn', type=str, default='GT', choices=['ELE', 'GT'],
                        help='数据集名称: ELE, GT')

    # 特征组合参数
    parser.add_argument('--fs_combine', type=str, default='base', choices=['eeg', 'et', 'et+eeg', 'base', "beta", "gamma", "beta+gamma"],
                        help='特征组合方式: eeg (脑电), et (眼动), et+eeg (全部), base (不加特征)')
    # OOV参数
    parser.add_argument('--OOV', type=str, default='yan', choices=['zero', 'yan', 'zhang', 'yan+zhang'],
                        help='有zero, yan, zhang, masor几种')
    # 模型参数
    parser.add_argument('--bert_weight', type=str, default='bert-base-uncased',
                        help='BERT模型名称')
    parser.add_argument('--ele_train_path', type=str, default='/users/li4xy/AKE/datas/Election_Trec/train.json')
    parser.add_argument('--gt_train_path', type=str, default='/users/li4xy/AKE/datas/General_Twitter/train.json')
    parser.add_argument('--ele_test_path', type=str, default='/users/li4xy/AKE/datas/Election_Trec/test.json')
    parser.add_argument('--gt_test_path', type=str, default='/users/li4xy/AKE/datas/General_Twitter/test.json')
    parser.add_argument('--bnc_path', type=str, default='/users/li4xy/AKE/datas/lemma.al')
    parser.add_argument('--model_path', type=str, default='./best_model.pt', help='保存模型路径')
    parser.add_argument('--max_len', type=int, default=512, help='最大序列长度')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=15,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批处理大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout概率')

    # 评估参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--test_only', action='store_true',
                        help='仅进行测试，不训练模型')
    parser.add_argument('--use_fasttext', action='store_true',
                        help='是否使用FastText处理OOV词')

    # 日志参数
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志目录')

    args = parser.parse_args()

    # 设置随机种子以便于重现结果
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)

    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"{args.dn}_{args.fs_combine}_{timestamp}.log")
    logger = setup_logger(log_file)

    # 记录参数配置
    logger.info(f"数据集: {args.dn}")
    logger.info(f"特征组合: {args.fs_combine}")
    logger.info(f"BERT模型: {args.bert_weight}")
    logger.info(f"批大小: {args.batch_size}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"训练轮数: {args.epochs}")

    # 加载数据
    logger.info("开始加载数据...")
    train_sens, train_Feature, train_tags, test_sens, test_Feature, test_tags = load_data(args)

    # 初始化分词器
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_weight)

    # 标签映射
    labels_to_ids = {'none': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4, "O": 5}
    ids_to_labels = {0: 'none', 1: 'B', 2: 'I', 3: 'E', 4: 'S', 5: 'O'}

    # 创建数据集
    logger.info("准备训练集...")
    train_dataset = MyDataset(train_sens, train_Feature, train_tags)
    train_dataset.encode(tokenizer, args.max_len, labels_to_ids)

    logger.info("准备测试集...")
    test_dataset = MyDataset(test_sens, test_Feature, test_tags)
    test_dataset.encode(tokenizer, args.max_len, labels_to_ids)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 合并认知字典
    logger.info("合并认知特征字典...")
    vocab_feature_dict = merge_cognitive_dicts(train_cognitive_dict, test_cognitive_dict)
    oov_feature_dict = merge_cognitive_dicts(train_oov_dict, test_oov_dict)

    # 准备 FastText（如果启用）
    if args.use_fasttext:
        logger.info("准备FastText模型和词向量...")
        fasttext_dict = prepare_fasttext(train_dataloader, test_dataloader, tokenizer, train_oov_dict,
                                         train_cognitive_dict)

    # 特征处理 - 根据参数选择使用的特征类型
    fs_num = 0
    if args.fs_combine == 'eeg':
        fs_num = 8  # 只使用脑电特征
        logger.info("使用脑电特征 (8维)")
    elif args.fs_combine == 'et':
        fs_num = 17  # 只使用眼动特征
        logger.info("使用眼动特征 (17维)")
    elif args.fs_combine == 'all':
        fs_num = 25  # 使用所有特征
        logger.info("使用全部特征 (25维)")
    elif args.fs_combine == 'none':
        fs_num = 0  # 不使用特征
        logger.info("不使用认知特征")

    # 初始化模型
    logger.info(f"初始化BERT模型 ({args.bert_weight})...")
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_weight)
    torch.manual_seed(1)
    model = BertNerModel(
        bert_weight=args.bert_weight,
        num_labels=6,
        fs_num=fs_num,
        tokenizer=tokenizer,
        vocab_feature_dict=vocab_feature_dict,  # 或 test_cognitive_dict
        oov_feature_dict=oov_feature_dict,  # 或 test_oov_dict
        fasttext_dict=fasttext_dict,
        fs_combine=args.fs_combine
    ).to(device)

    # 如果仅进行测试
    if args.test_only:
        logger.info("加载已有模型进行测试...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        P, R, F1 = test_model(model, test_dataloader, device, logger)
        logger.info(f"测试结果: 精确率={P:.4f}, 召回率={R:.4f}, F1分数={F1:.4f}")
    else:
        # 训练和评估模型
        logger.info("开始训练模型...")
        best_f1 = train_model(model, train_dataloader, test_dataloader, args, device, logger)

        # 加载最佳模型进行最终测试
        logger.info(f"使用最佳模型进行最终测试 (最佳F1: {best_f1:.4f})...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        P, R, F1 = test_model(model, test_dataloader, device, logger)

        # 记录最终结果
        logger.info(f"最终测试结果: 精确率={P:.4f}, 召回率={R:.4f}, F1分数={F1:.4f}")

    logger.info("程序执行完成")


if __name__ == "__main__":
    main()