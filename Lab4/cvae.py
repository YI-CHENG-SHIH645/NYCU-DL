import time
import json
import string
import argparse
import numpy as np
import pandas as pd
import random
import copy
from itertools import product, repeat
from collections import defaultdict
from os import path
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from multiprocessing import cpu_count
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import torch
from torch import nn
import matplotlib.pyplot as plt

TRAINING_DATA_PATH = './Lab4 - dataset/train.txt'
TESTING_DATA_PATH = './Lab4 - dataset/test.txt'
TENSES = ['sp', 'tp', 'pg', 'p']
TESTING_TENSE_CONVERSION_PAIRS = [['sp', 'p'], ['sp', 'pg'], ['sp', 'tp'],
                                  ['sp', 'tp'], ['p', 'tp'], ['sp', 'pg'],
                                  ['p', 'sp'], ['pg', 'sp'], ['pg', 'p'],
                                  ['pg', 'tp']]

ITERATION = 500
WORDS_PER_ITER = 4908
TEACHER_START = 150 * WORDS_PER_ITER
KL_START = 70 * WORDS_PER_ITER
KL_ANNEALING_METHOD = 'Cyclic'
EXPERIMENT_NAME = KL_ANNEALING_METHOD
Path(EXPERIMENT_NAME).mkdir(parents=True, exist_ok=True)
PATH = {
    'training_process': path.join(EXPERIMENT_NAME, '{}.png'),
    'model_weights': path.join(EXPERIMENT_NAME, '{}.pth')
}


# ------------------------------------------ Dataset ------------------------------------------
class WordPairDataset(Dataset):
    def __init__(self, data_dir: str, mode: str, max_length,
                 tenses: list = None, pad=False):
        assert mode in ['train', 'test']
        self.mode = mode
        self.tenses = tenses
        self.pad = pad
        self.max_length = max_length
        self.data_dir = data_dir
        self.raw_data_file = mode + '.txt'
        self.raw_data_path = path.join(data_dir, self.raw_data_file)

        self.character_file = 'character.json'
        self.data_file = mode + '.json'

        if path.exists(path.join(data_dir, self.data_file)):
            self._load_data()
            self._load_character()
        else:
            if not path.exists(path.join(data_dir, self.character_file)):
                self._create_character()
                self._load_character()
            else:
                self._load_character()
            self._create_data()
            self._load_data()
        # self.data = {k: v for k, v in list(self.data.items())[:200]}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input_seq': np.asarray(self.data[idx]['input']),
            'input_tense': self.data[idx]['inp_tense'],
            'target_seq': np.asarray(self.data[idx]['target']),
            'target_tense': self.data[idx]['tgt_tense'],
            'input_length': self.data[idx]['inp_length'],
            'target_length': self.data[idx]['tgt_length']
        }

    @property
    def pad_idx(self):
        return self.w2i['<pad>'] if self.pad else None

    @property
    def sow_idx(self):
        return self.w2i['<sow>']

    @property
    def eow_idx(self):
        return self.w2i['<eow>']

    @property
    def vocab_size(self):
        return len(self.w2i)

    def _create_data(self):
        data = defaultdict(dict)
        with open(path.join(self.data_dir, self.raw_data_file), 'r') as raw_data:
            for i, line in enumerate(raw_data):
                vocabs = line.strip().split(' ')
                vocabs_inps, vocabs_tgts = [], []
                # fixed pattern at each line if train
                tenses = self.tenses if self.mode == 'train' else self.tenses[i]
                for vocab, tense in zip(vocabs, tenses):
                    characters = list(vocab)

                    inp = characters[:self.max_length - 1]
                    inp = inp + ['<eow>']

                    tgt = characters[:self.max_length - 1]
                    tgt = tgt + ['<eow>']

                    inp_length = len(inp)
                    tgt_length = len(tgt)

                    if self.pad:
                        inp.extend(['<pad>'] * (self.max_length - len(inp)))
                        tgt.extend(['<pad>'] * (self.max_length - len(tgt)))

                    inp = ([self.w2i[c] for c in inp], TENSES.index(tense), inp_length)
                    tgt = ([self.w2i[c] for c in tgt], TENSES.index(tense), tgt_length)

                    vocabs_inps.append(inp)
                    vocabs_tgts.append(tgt)
                if self.mode == 'train':
                    # inp_tgt_combs = product(vocabs_inps, vocabs_tgts)
                    inp_tgt_combs = zip(vocabs_inps, vocabs_tgts)
                else:
                    inp_tgt_combs = [(vocabs_inps[0], vocabs_tgts[1])]
                for (inp, inp_tense, inp_l), (tgt, tgt_tense, tgt_l) in inp_tgt_combs:
                    key = len(data)
                    data[key]['input'] = inp
                    data[key]['inp_tense'] = inp_tense
                    data[key]['target'] = tgt
                    data[key]['tgt_tense'] = tgt_tense
                    data[key]['inp_length'] = inp_l
                    data[key]['tgt_length'] = tgt_l

            with open(path.join(self.data_dir, self.data_file), 'wb') as data_file:
                data = json.dumps(data, ensure_ascii=False)
                data_file.write(data.encode('utf8', 'replace'))

    def _load_data(self):
        with open(path.join(self.data_dir, self.data_file), 'r') as data_file:
            self.data = json.load(data_file)

    def _create_character(self):
        assert self.mode == 'train'

        w2i = dict()
        i2w = dict()

        special_character = ['<sow>', '<eow>']
        if self.pad:
            special_character = ['<pad>'] + special_character
        for sc in special_character:
            i2w[len(w2i)] = sc
            w2i[sc] = len(w2i)

        for c in string.ascii_lowercase:
            i2w[len(w2i)] = c
            w2i[c] = len(w2i)

        characters = dict(w2i=w2i, i2w=i2w)
        with open(path.join(self.data_dir, self.character_file), 'wb') as ch_file:
            data = json.dumps(characters, ensure_ascii=False)
            ch_file.write(data.encode('utf8', 'replace'))

    def _load_character(self):
        with open(path.join(self.data_dir, self.character_file), 'r') as ch_file:
            character = json.load(ch_file)

        self.w2i, self.i2w = character['w2i'], character['i2w']


# ------------------------------------------ Dataset ------------------------------------------


# ------------------------------------------ Model ------------------------------------------
# rnn_type, word_dropout, embedding_dropout, num_layers, bidirectional
class LstmCVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                 con_embedding_size, hidden_size, latent_size,
                 pad_idx, sow_idx, eow_idx, i2w, batch_size, device):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.con_embedding_size = con_embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.pad_idx = pad_idx
        self.sow_idx = sow_idx
        self.eow_idx = eow_idx
        self.i2w = i2w
        self.batch_size = batch_size
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.con_embedding = nn.Embedding(len(TENSES), con_embedding_size)
        self.en_lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)

        self.hidden2mu = nn.Linear(hidden_size, latent_size)
        self.hidden2logvar = nn.Linear(hidden_size, latent_size)

        self.cell2mu = nn.Linear(hidden_size, latent_size)
        self.cell2logvar = nn.Linear(hidden_size, latent_size)

        self.hidden_latent2rnn = nn.Linear(latent_size + con_embedding_size, hidden_size)
        self.cell_latent2rnn = nn.Linear(latent_size + con_embedding_size, hidden_size)

        self.de_lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.de_out2vocab = nn.Linear(hidden_size, vocab_size)

        # where to use this ?
        # self.dropout = nn.Dropout(p=0.5)

    def init_hidden(self, condition, hidden_z=None, cell_z=None):
        batch_size = condition.size(0)
        # 1 x batch_size x con_em_size
        con_embedded = self.con_embedding(condition).view(1, batch_size, -1)
        if hidden_z is None:
            hidden = torch.zeros(1, batch_size, self.hidden_size - self.con_embedding_size).to(self.device)
            cell = torch.zeros(1, batch_size, self.hidden_size - self.con_embedding_size).to(self.device)
            return torch.cat([hidden, con_embedded], dim=-1), torch.cat([cell, con_embedded], dim=-1)
        else:
            assert cell_z is not None
            return self.hidden_latent2rnn(torch.cat([hidden_z, con_embedded], dim=-1)), \
                self.cell_latent2rnn(torch.cat([cell_z, con_embedded], dim=-1))

    def encode(self, x, hidden):
        out, hidden = self.en_lstm(x, hidden)
        return out, hidden

    def reparameterize(self, hidden_mu, hidden_logvar, cell_mu, cell_logvar):
        hidden_std = torch.exp(0.5 * hidden_logvar)
        hidden_gaussian_noise = torch.randn_like(hidden_std).to(self.device)
        hidden_z = hidden_mu + hidden_gaussian_noise * hidden_std

        cell_std = torch.exp(0.5 * cell_logvar)
        cell_gaussian_noise = torch.randn_like(cell_std).to(self.device)
        cell_z = cell_mu + cell_gaussian_noise * cell_std

        return hidden_z, cell_z

    def decode(self, x, hidden):
        # decoder 在 testing phase 用到，batch_size 可能不同
        batch_size = x.size(0)
        x = self.embedding(x).view(batch_size, 1, -1)
        out, hidden = self.de_lstm(x, hidden)
        return self.de_out2vocab(out.view(batch_size, -1)), hidden

    def forward(self,
                input_seq, input_tense,
                target_seq, target_tense,
                input_length, target_length,
                mode, iteration):

        # 凡使用 forward 都能確保 batch_size 正確設置，
        # decode、init_hidden（generate）之 batch_size 另行設定
        batch_size = input_seq.size(0)
        input_length, sorted_idx = torch.sort(input_length, descending=True)

        # batch_size x max_length
        input_seq = input_seq[sorted_idx]
        input_tense = input_tense[sorted_idx]
        target_seq = target_seq[sorted_idx]
        target_tense = target_tense[sorted_idx]
        target_length = target_length[sorted_idx]

        # 1 x batch_size x hidden_size
        hidden = self.init_hidden(input_tense)

        # batch_size x max_length x embedding_size
        input_seq = self.embedding(input_seq)
        packed_input = nn.utils.rnn.pack_padded_sequence(input_seq, input_length.cpu(), batch_first=True)

        # 1 x batch_size x hidden_size
        _, hidden = self.encode(packed_input, hidden)

        h_n, c_n = hidden

        # 1 x batch_size x latent_size
        h_mu = self.hidden2mu(h_n)
        h_logvar = self.hidden2logvar(h_n)
        c_mu = self.cell2mu(c_n)
        c_logvar = self.cell2logvar(c_n)

        # 1 x batch_size x latent_size
        hidden_z, cell_z = self.reparameterize(h_mu, h_logvar, c_mu, c_logvar)

        raw_out_foreach_word, model_predictions = [], []
        use_teacher_forcing = random.random() < teacher_forcing_scheduler(iteration) if mode == 'train' else False
        decoder_input = torch.tensor([self.sow_idx] * batch_size).to(self.device)

        # 1 x batch_size x hidden_size
        hidden = self.init_hidden(target_tense, hidden_z=hidden_z, cell_z=cell_z)
        # hidden = self.dropout(hidden[0]), self.dropout(hidden[1])

        # 實驗顯示
        # batch_size has to be 1
        # batch_size has to be 1
        # batch_size has to be 1
        # batch_size=1 we can change max_length -> target_length.item()
        for di in range(target_length.item()):
            # out: batch_size x vocab_size
            out, hidden = self.decode(decoder_input, hidden)

            raw_out_foreach_word.append(out)
            # batch_size,
            model_idx_ch_prediction = torch.argmax(out, dim=-1)
            if use_teacher_forcing:
                decoder_input = target_seq[:, di]
            else:
                decoder_input = model_idx_ch_prediction

            model_predictions.append(model_idx_ch_prediction.cpu().numpy())
        # model_predictions: batch_size x max_length
        model_predictions = np.asarray(model_predictions).T
        str_model_predictions = [idx_ch_arr_to_string(idx_ch_arr, self.i2w, self.eow_idx)
                                 for idx_ch_arr in model_predictions]
        str_target_seq = [idx_ch_arr_to_string(idx_ch_arr, self.i2w, self.eow_idx)
                          for idx_ch_arr in target_seq.cpu().numpy()]
        # (max_length x batch_size) x vocab_size
        return torch.vstack(raw_out_foreach_word), \
               {'mu': h_mu.squeeze(0), 'logvar': h_logvar.squeeze(0),
                'mu_c': c_mu.squeeze(0), 'logvar_c': c_logvar.squeeze(0)}, \
               list(zip(str_model_predictions, str_target_seq))


# ------------------------------------------ Model ------------------------------------------


# ------------------------------------------ Scheduler / Metrics ------------------------------------------
def loss_fn(criterion, raw_out, tgt, tgt_length,
            mu, logvar, mu_c, logvar_c):
    # train a word at a time, no padding, trim padding away
    bce = criterion(raw_out, tgt.T.ravel()[:tgt_length])
    hidden_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    cell_kld = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())
    kld = (hidden_kld + cell_kld) / 2.

    return bce, kld


def teacher_forcing_scheduler(nth_iter):
    return 1 if nth_iter <= TEACHER_START \
        else 1 - ((nth_iter - TEACHER_START) / (ITERATION * WORDS_PER_ITER - TEACHER_START)) / 2


def kl_weight_scheduler(nth_iter, method='Monotonic'):
    assert method in ['Monotonic', 'Cyclic']
    if method == 'Monotonic':
        return 0 if (nth_iter / KL_START) <= 1 \
            else (nth_iter - KL_START) / (ITERATION * WORDS_PER_ITER - KL_START) * 0.8
    else:
        return (nth_iter % KL_START) / KL_START


def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference], output, weights=weights, smoothing_function=cc.method1)


# 產生 100 套時態字，有幾 % 是出現在 train.txt 中的 (以套計算)
def gaussian_score(words):
    assert words.shape == (100, 4)
    words = words.tolist()
    words_list = []
    score = 0
    with open(TRAINING_DATA_PATH, 'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score / len(words)
    # words = words.tolist()
    # score = 0
    # with open(TRAINING_DATA_PATH, 'r') as fp:
    #     training_words = [line.strip().split(' ') for line in fp.readlines()]
    #     for word in words:
    #         for t_word in training_words:
    #             score += word == t_word
    # return score / len(words)


def gaussian_noise_generate_tenses(cvae: LstmCVAE, print_result=False):
    words_with_tenses = []

    # 100 different latent z for each word
    hidden_z = torch.randn(1, 100, cvae.latent_size).to(cvae.device)
    cell_z = torch.randn(1, 100, cvae.latent_size).to(cvae.device)

    for tense in range(len(TENSES)):
        condition = torch.tensor([tense] * 100).to(cvae.device)
        # 1 x 100 x hidden_size
        all_hidden = cvae.init_hidden(condition, hidden_z=hidden_z, cell_z=cell_z)
        ch_with_tense, idx_ch_arrays = [], []
        for i in range(100):
            # 1 x 1 x hidden_size
            hidden = all_hidden[0][:, i:i + 1, :], all_hidden[1][:, i:i + 1, :]
            decoder_input = torch.tensor([cvae.sow_idx]).to(cvae.device)
            # a word each time, generate max length 20
            for _ in range(20):
                # out: 1 x 1 x vocab_size
                out, hidden = cvae.decode(decoder_input, hidden)

                # pred_idx: 1,
                pred_idx = torch.argmax(out, dim=-1)
                ch_with_tense.append(pred_idx.cpu().numpy()[0])
                decoder_input = pred_idx

                if pred_idx.item() == cvae.eow_idx:
                    break
            # 1, generated_length
            idx_ch_arrays.append(np.asarray(ch_with_tense))
            ch_with_tense.clear()
        words_with_tenses.append([idx_ch_arr_to_string(idx_ch_arr, cvae.i2w, cvae.eow_idx)
                                  for idx_ch_arr in idx_ch_arrays])

    # 100, 4
    words_with_tenses = np.asarray(words_with_tenses).T
    score = gaussian_score(words_with_tenses)
    if print_result:
        print(words_with_tenses[-15:])
        print(f'Gaussian-score: {score:.2f}')
    return score


def eval_test_bleu(cvae: LstmCVAE, test_loader, device, print_result=False):
    all_word_pairs, input_words = [], []
    for data in test_loader:
        input_words.extend([idx_ch_arr_to_string(data['input_seq'][0].numpy(),
                            test_loader.dataset.i2w,
                            test_loader.dataset.eow_idx)])
        data = to_device(device, **data)
        _, _, word_pairs = cvae(mode='test', iteration=0, **data)
        all_word_pairs.extend(word_pairs)
    score = sum(compute_bleu(*pair) for pair in all_word_pairs) / len(all_word_pairs)
    if print_result:
        for i, input_word in enumerate(input_words):
            print(f'Input: {input_word: <20}'
                  f'Target: {all_word_pairs[i][1]: <20}'
                  f'Prediction: {all_word_pairs[i][0]: <20}')
        print(f'BLEU4-score: {score:.2f} \n')
    return score


def idx_ch_arr_to_string(idx_ch_arr, i2w: dict, eow_idx: int):
    for i, idx_ch in enumerate(idx_ch_arr):
        if idx_ch == eow_idx:
            return ''.join([i2w[str(idx)] for idx in idx_ch_arr[:i]])
    return ''.join([i2w[str(idx)] for idx in idx_ch_arr])


def plot_history(history: pd.DataFrame):
    plt.figure(figsize=(16, 9))
    ax = history.reset_index().plot(x='index', y='BCE', color='orange', label='BCE')
    history.reset_index().plot(x='index', y='KLD', color='blue', ax=ax, label='KLD')
    ax2 = ax.twinx()
    history.reset_index().plot(x='index', y='Teacher_ratio', style='c--',
                               ax=ax2, label='teacher_ratio')
    history.reset_index().plot(x='index', y='KLD_weight', style='r--',
                               ax=ax2, label='KLD_weight')
    history.reset_index().plot.scatter(x='index', y='BLEU4-score', color='green',
                                       ax=ax2, label='BLEU4-score', s=4)
    history.reset_index().plot.scatter(x='index', y='Gaussian-score', color='brown',
                                       ax=ax2, label='Gaussian-score', s=4)
    ax2.get_legend().remove()
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc=0)
    ax.set_xlabel("500 iterations")
    ax.set_ylabel("loss")
    ax2.set_ylabel("score / weight")
    plt.title(f'Training loss/ratio curve ({KL_ANNEALING_METHOD})')
    plt.savefig(PATH['training_process'].format(EXPERIMENT_NAME))


# ------------------------------------------ Scheduler / Metrics ------------------------------------------


# ------------------------------------------ Process ------------------------------------------
def to_device(device, **kwargs):
    kwargs['input_seq'] = kwargs['input_seq'].to(device)
    kwargs['target_seq'] = kwargs['target_seq'].to(device)
    kwargs['input_tense'] = kwargs['input_tense'].to(device)
    kwargs['target_tense'] = kwargs['target_tense'].to(device)

    return kwargs


def train(model, optimizer, criterion, iteration, ith_word, **kwargs):
    iteration = iteration * WORDS_PER_ITER + ith_word
    optimizer.zero_grad()
    raw_out, mu_logvar, word_pairs = model(mode='train', iteration=iteration, **kwargs)
    bce, kld = loss_fn(criterion, raw_out, kwargs['target_seq'], kwargs['target_length'].item(), **mu_logvar)
    kld_weight = kl_weight_scheduler(iteration, method=KL_ANNEALING_METHOD)
    loss = bce + kld_weight * kld
    loss.backward()
    optimizer.step()

    return bce, kld / 2, word_pairs


def main(args):
    mode = 'test' if args.test else 'train'

    data_loader = None
    if not args.test:
        dataset = WordPairDataset(data_dir="./Lab4 - dataset", mode='train',
                                  max_length=20, tenses=TENSES, pad=True)
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=not args.test,
                                 num_workers=cpu_count(),
                                 pin_memory=torch.cuda.is_available())

    test_dataset = WordPairDataset(data_dir="./Lab4 - dataset", mode='test',
                                   max_length=20, tenses=TESTING_TENSE_CONVERSION_PAIRS, pad=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_params = dict(
        pad_idx=test_dataset.pad_idx,
        sow_idx=test_dataset.sow_idx,
        eow_idx=test_dataset.eow_idx,
        vocab_size=test_dataset.vocab_size,
        i2w=test_dataset.i2w,
        embedding_size=256,
        con_embedding_size=8,
        hidden_size=256,
        latent_size=32,
        batch_size=1,
        device=device
    )
    model = LstmCVAE(**model_params).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = MultiStepLR(optimizer, milestones=[71], gamma=1)
    criterion = nn.CrossEntropyLoss()

    if args.test:
        model.load_state_dict(torch.load(PATH['model_weights'].format('aa0.72_0.45_142')))
        model.eval()
        eval_test_bleu(model, test_loader, device, print_result=True)
        gaussian_noise_generate_tenses(model, print_result=True)
        return

    def repeater(dl):
        for loader in repeat(dl):
            for data in loader:
                yield data

    data_loader = repeater(data_loader)

    history = []
    stats_to_track = ['bce', 'kld']
    stats = defaultdict(float)
    for iteration in range(ITERATION):
        print(f'Iteration: {iteration + 1}')

        total_bce, total_kld = 0, 0
        all_word_pairs = []

        model.train()
        # iter time
        t0 = time.time()
        for nth_word in range(WORDS_PER_ITER):

            batch = to_device(device, **next(data_loader))
            bce, kld, word_pairs = train(model, optimizer, criterion, iteration, nth_word, **batch)
            for stat in stats_to_track:
                stats[stat] += locals()[stat].item()
            all_word_pairs.extend(word_pairs)
        print(f'iteration time: {time.time() - t0:.1f} s')

        # end of an iteration
        for stat in stats_to_track:
            stats[stat] /= WORDS_PER_ITER

        model.eval()
        gau_score = gaussian_noise_generate_tenses(model, print_result=True)
        bleu4 = sum(compute_bleu(*pair) for pair in all_word_pairs) / len(all_word_pairs)
        test_bleu4 = eval_test_bleu(cvae=model, test_loader=test_loader, device=device)

        teacher_ratio = teacher_forcing_scheduler(iteration * WORDS_PER_ITER)
        kl_weight = kl_weight_scheduler(iteration * WORDS_PER_ITER, method=KL_ANNEALING_METHOD)

        if test_bleu4 >= 0.6 and gau_score >= 0.1:
            torch.save(copy.deepcopy(model.state_dict()),
                       PATH['model_weights'].format(f'{test_bleu4:.2f}_{gau_score:.2f}_{iteration+1}'))

        history.append([stats['bce'], stats['kld']] + [test_bleu4, gau_score, teacher_ratio, kl_weight])
        print(f'bce: {stats["bce"]:.3f},  kld: {stats["kld"]:.3f},  '
              f'BLEU4: {bleu4:.3f},  test_BLEU4: {test_bleu4:.3f},  '
              f'gau: {gau_score:.3f},  kl_w: {kl_weight:.3f},  teacher: {teacher_ratio:.3f}')
        print('\n\n')
        stats.clear()
        scheduler.step()
        if iteration + 1 == 300:
            break
    history = pd.DataFrame(history,
                           columns=['BCE', 'KLD', 'BLEU4-score', 'Gaussian-score',
                                    'Teacher_ratio', 'KLD_weight'],
                           index=range(1, len(history) + 1))
    plot_history(history)


# ------------------------------------------ Process ------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False)
    arguments = parser.parse_args()

    main(arguments)
