import abc
import csv
import torch
from dataclasses import dataclass, astuple
from os import path
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer


@dataclass
class Instance:
    sentence1: str
    sentence2: str
    gold_label: str


class DataPrep(abc.ABC):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    @abc.abstractmethod
    def labels(self):
        raise NotImplementedError

    @staticmethod
    def acc_func():
        def f(outputs, labels):
            y_pred = np.argmax(outputs, axis=1).ravel()
            return accuracy_score(labels, y_pred)
        return f

    def pred_str(self, outputs):
        str_labels = self.labels()
        return [str_labels[int_labels] for int_labels in np.argmax(outputs, axis=1).tolist()]

    @staticmethod
    def _read_tsv(file_path):
        with open(file_path, 'r') as f:
            lines = csv.reader(f, delimiter='\t')
            return [line for line in lines]


class MNLIPrep(DataPrep):
    def labels(self):
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _prep(lines: list):
        instances = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            sentence_1 = line[8]
            sentence_2 = line[9]
            gold_label = line[-1]
            instances.append(Instance(sentence_1, sentence_2, gold_label))
        return instances

    def create_bert_input_dict(self, split: str, instance_slice: slice, return_tokens=False):
        instances = self._prep(self._read_tsv(path.join(self.data_dir, f'{split}.tsv')))[instance_slice]
        batch_instances = list(zip(*map(astuple, instances)))
        batch_sentences = batch_instances[:2]
        batch_gold_labels = batch_instances[-1]
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', max_length=128)
        bert_input = tokenizer(*batch_sentences, padding='max_length', return_tensors='pt', return_length=True)
        lengths = bert_input.pop('length')
        all_tokens = []
        for i in range(len(bert_input['input_ids'])):
            token = tokenizer.convert_ids_to_tokens(bert_input['input_ids'][i])
            all_tokens.append(token[:lengths[i]])
        str_labels = self.labels()
        label_ids = np.array(list(map(str_labels.index, batch_gold_labels)))
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        bert_input.update({'label_ids': label_ids})

        if return_tokens:
            return bert_input, all_tokens
        return bert_input
