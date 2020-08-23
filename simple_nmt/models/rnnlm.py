import torch
import torch.nn as nn

import simple_nmt.data_loader as data_loader


class LanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        word_vec_size=256,
        hidden_size=512,
        n_layers=4,
        dropout_p=.3,
        max_length=255
    ):
        self.vocab_size = vocab_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        super().__init__()

        self.emb = nn.Embedding(
            vocab_size,
            word_vec_size,
            padding_idx=data_loader.PAD,
        )
        self.rnn = nn.LSTM(
            word_vec_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout_p,
        )
        self.out = nn.Linear(hidden_size, vocab_size, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, *args):
        # |x| = (batch_size, length)
        x = self.emb(x) 
        # |x| = (batch_size, length, word_vec_size)
        x, _ = self.rnn(x) 
        # |x| = (batch_size, length, hidden_size)
        x = self.out(x) 
        # |x| = (batch_size, length, vocab_size)
        y_hat = self.log_softmax(x)

        return y_hat

    def search(self, batch_size=64, max_length=255):
        x = torch.LongTensor(batch_size, 1).to(next(self.parameters()).device).zero_() + data_loader.BOS
        # |x| = (batch_size, 1)
        is_undone = x.new_ones(batch_size, 1).float()

        y_hats, indice = [], []
        h, c = None, None
        while is_undone.sum() > 0 and len(indice) < max_length:
            x = self.emb(x)
            # |emb_t| = (batch_size, 1, word_vec_size)

            x, (h, c) = self.rnn(x, (h, c)) if h is not None and c is not None else self.rnn(x)
            # |x| = (batch_size, 1, hidden_size)
            y_hat = self.log_softmax(x)
            # |y_hat| = (batch_size, 1, output_size)
            y_hats += [y_hat]

            # y = torch.topk(y_hat, 1, dim = -1)[1].squeeze(-1)
            y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1)
            y = y.masked_fill_((1. - is_undone).byte(), data_loader.PAD)
            is_undone = is_undone * torch.ne(y, data_loader.EOS).float()            
            # |y| = (batch_size, 1)
            # |is_undone| = (batch_size, 1)
            indice += [y]

            x = y

        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=-1)
        # |y_hat| = (batch_size, length, output_size)
        # |indice| = (batch_size, length)

        return y_hats, indice
