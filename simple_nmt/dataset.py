from argparse import Namespace

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from torchtext.vocab import build_vocab_from_iterator


SPECIAL_TOKENS = {
    'PAD': '<PAD>',
    'UNK': '<UNK>',
    'BOS': '<BOS>',
    'EOS': '<EOS>',
    'PAD_idx': 0,
    'UNK_idx': 1,
    'BOS_idx': 2,
    'EOS_idx': 3,
}
SPECIAL_TOKENS = Namespace(**SPECIAL_TOKENS)


def read_text(fn, exts, max_length=256):
    null_cnt = 0
    max_cnt = 0

    parallel_lines = []

    with open(fn + '.' + exts[0], 'r') as f1:
        with open(fn + '.' + exts[1], 'r') as f2:
            for l1, l2 in zip(f1.readlines(), f2.readlines()):
                if l1.strip() == '' or l2.strip() == '':
                    null_cnt += 1
                    continue
                if len(l1.split()) > max_length or len(l2.split()) > max_length:
                    max_cnt += 1
                    continue

                parallel_lines += [(
                    l1.strip().split(),
                    l2.strip().split(),
                )]

    print('Read %s.%s and %s.%s.' % (fn, exts[0], fn, exts[1]))
    print('Null data count: %d' % null_cnt)
    print('Exceeding maximum length data count: %d' % max_cnt)
    print('Available data count: %d' % len(parallel_lines))

    return parallel_lines


def get_vocab(texts, min_freq=1):
    vocab = build_vocab_from_iterator(
        texts,
        min_freq=min_freq,
        specials=[
            SPECIAL_TOKENS.PAD,
            SPECIAL_TOKENS.UNK,
            SPECIAL_TOKENS.BOS,
            SPECIAL_TOKENS.EOS,
        ],
        special_first=True
    )
    vocab.set_default_index(1)

    return vocab


class MachineTranslationDataset(Dataset):

    def __init__(self, texts, special_token_at_both=False):
        self.texts = texts
        self.special_token_at_both = special_token_at_both
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = [
            self.texts[item][0],
            self.texts[item][1],
        ]

        if self.special_token_at_both:
            text[0] = [SPECIAL_TOKENS.BOS] + text[0] + [SPECIAL_TOKENS.EOS]
        text[1] = [SPECIAL_TOKENS.BOS] + text[1] + [SPECIAL_TOKENS.EOS]

        return {
            'src': text[0],
            'tgt': text[1],
        }


class MachineTranslationCollator():

    def __init__(self, src_vocab, tgt_vocab):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __call__(self, samples):
        src = sorted(
            [(torch.tensor(
                self.src_vocab(s['src']),
                dtype=torch.long
             ), len(s['src'])) for s in samples],
            key=lambda x: x[1],
            reverse=True,
        )
        tgt = sorted(
            [(torch.tensor(
                self.tgt_vocab(s['tgt']),
                dtype=torch.long
             ), len(s['tgt'])) for s in samples],
            key=lambda x: x[1],
            reverse=True,
        )

        return_value = {
            'src': (
                pad_sequence(
                    [s[0] for s in src],
                    batch_first=True,
                    padding_value=0,
                ),
                torch.tensor(
                    [s[1] for s in src],
                    dtype=torch.long
                )
            ),
            'tgt': (
                pad_sequence(
                    [t[0] for t in tgt],
                    batch_first=True,
                    padding_value=0,
                ),
                torch.tensor(
                    [t[1] for t in tgt],
                    dtype=torch.long
                )
            ),
        }

        # print('src')
        # print(return_value['src'][0].shape)
        # print(return_value['src'][0])
        # print(return_value['src'][1])
        # print('tgt')
        # print(return_value['tgt'][0].shape)
        # print(return_value['tgt'][0])
        # print(return_value['tgt'][1])

        return Namespace(**return_value)
