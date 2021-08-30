import random
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
    vocab.set_default_index(SPECIAL_TOKENS.UNK_idx)

    return vocab


class MachineTranslationDataset(Dataset):

    def __init__(self, texts, src_vocab, tgt_vocab, special_token_at_both=False):
        self.texts = texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.special_token_at_both = special_token_at_both

        for i in range(len(texts)):
            if self.special_token_at_both:
                src_text = [SPECIAL_TOKENS.BOS] + texts[i][0] + [SPECIAL_TOKENS.EOS]
            else:
                src_text = texts[i][0]
            tgt_text = [SPECIAL_TOKENS.BOS] + texts[i][1] + [SPECIAL_TOKENS.EOS]

            texts[i] = (
                torch.tensor(src_vocab(src_text), dtype=torch.long),
                torch.tensor(tgt_vocab(tgt_text), dtype=torch.long),
            )
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {
            'src': self.texts[item][0],
            'tgt': self.texts[item][1],
        }


class MachineTranslationCollator():

    def __call__(self, samples):
        src = sorted(
            [(s['src'], s['src'].size(-1)) for s in samples],
            key=lambda x: x[1],
            reverse=True,
        )
        tgt = sorted(
            [(s['tgt'], s['tgt'].size(-1)) for s in samples],
            key=lambda x: x[1],
            reverse=True,
        )

        return_value = {
            'src': (
                pad_sequence(
                    [s[0] for s in src],
                    batch_first=True,
                    padding_value=SPECIAL_TOKENS.PAD_idx,
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
                    padding_value=SPECIAL_TOKENS.PAD_idx,
                ),
                torch.tensor(
                    [t[1] for t in tgt],
                    dtype=torch.long
                )
            ),
        }

        return Namespace(**return_value)


class SequenceLengthBasedBatchSampler():

    def __init__(self, texts, batch_size):
        self.batch_size = batch_size
        self.lens = [len(s[1]) for s in texts]
        self.indice = [i for i in range(len(texts))]

        tmp = sorted(zip(self.lens, self.indice), key=lambda x: x[0])
        self.sorted_lens = [x[0] for x in tmp]
        self.sorted_indice = [x[1] for x in tmp]

    def __iter__(self):
        batch_indice = [i for i in range(0, len(self.lens), self.batch_size)]
        random.shuffle(batch_indice)

        for i, batch_idx in enumerate(batch_indice):
            ret = self.sorted_indice[batch_idx:batch_idx + self.batch_size]

            yield ret
        
    def __len__(self):
        return len(self.lens) // self.batch_size
