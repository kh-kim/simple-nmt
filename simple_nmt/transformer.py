import torch
import torch.nn as nn

import data_loader


class Attention(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, dk=64):
        # |Q| = (batch_size, m, hidden_size)
        # |K| = |V| = (batch_size, n, hidden_size)

        w = torch.bmm(Q, K.transpose(1, 2))
        # |w| = (batch_size, m, n)
        if mask is not None:
            assert w.size() == mask.size()
            w.masked_fill_(mask, -float('inf'))

        w = self.softmax(w / (dk**.5))
        c = torch.bmm(w, V)
        # |c| = (batch_size, m, hidden_size)

        return c


class MultiHead(nn.Module):
    
    def __init__(self, hidden_size, n_splits):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_splits = n_splits

        self.Q_linear = nn.Linear(hidden_size, hidden_size)
        self.K_linear = nn.Linear(hidden_size, hidden_size)
        self.V_linear = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

        self.attn = Attention()

    def forward(self, Q, K, V, mask=None):
        # |Q| = (batch_size, m, hidden_size)
        # |K| = |V| = (batch_size, n, hidden_size)
        # |mask| = (batch_size, m, n)

        QWs = self.Q_linear(Q).split(self.hidden_size // self.n_splits, dim=-1)
        KWs = self.K_linear(K).split(self.hidden_size // self.n_splits, dim=-1)
        VWs = self.V_linear(V).split(self.hidden_size // self.n_splits, dim=-1)
        # |QW_i| = (batch_size, m, hidden_size / n_splits)
        # |KW_i| = |VW_i| = (batch_size, n, hidden_size / n_splits)

        QWs = torch.cat(QWs, dim=0)
        KWs = torch.cat(KWs, dim=0)
        VWs = torch.cat(VWs, dim=0)
        # |QWs| = (batch_size * n_splits, m, hidden_size / n_splits)
        # |KWs| = |VWs| = (batch_size * n_splits, n, hidden_size / n_splits)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.n_splits)], dim=0)
            # |mask| = (batch_size * n_splits, m, n)

        c = self.attn(
            QWs, KWs, VWs,
            mask=mask,
            dk=self.hidden_size / self.n_splits,
        )
        # |c| = (batch_size * n_splits, m, hidden_size / n_splits)
        c = c.split(Q.size(0), dim=0)
        # |c_i| = (batch_size, m, hidden_size / n_splits)
        c = self.linear(torch.cat(c, dim=-1))
        # |c| = (batch_size, m, hidden_size)

        return c


class EncoderBlock(nn.Module):
    
    def __init__(self, hidden_size, n_splits, dropout_p=.1):
        super().__init__()

        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        # |x| = (batch_size, n, hidden_size)
        # |mask| = (batch_size, n, n)

        z = self.attn_norm(x + self.attn_dropout(self.attn(x, x, x, mask=mask)))
        z = self.fc_norm(z + self.fc_dropout(self.fc(z)))
        # |z| = (batch_size, n, hidden_size)

        return z, mask

class DecoderBlock(nn.Module):
    
    def __init__(self, hidden_size, n_splits, dropout_p=.1):
        super().__init__()

        self.masked_attn = MultiHead(hidden_size, n_splits)
        self.masked_attn_norm = nn.LayerNorm(hidden_size)
        self.masked_attn_dropout = nn.Dropout(dropout_p)

        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, KV, mask, prev):
        if prev is not None:
            # |x| = (batch_size, m=1, hidden_size)
            # |prev_i| = (batch_size, m', hidden_size)

            z = self.masked_attn_norm(x + self.masked_attn_dropout(
                self.masked_attn(x, prev, prev, mask=None)
            ))
            # |z| = (batch_size, 1, hidden_size)
        else:
            # |x| = (batch_size, m, hidden_size)
            batch_size = x.size(0)
            m = x.size(1)

            forward_mask = torch.triu(x.new_ones((m, m)), diagonal=1).byte()
            # |forward_mask| = (m, m)
            forward_mask = forward_mask.unsqueeze(0).expand(batch_size, *forward_mask.size())
            # |forward_mask| = (batch_size, m, m)

            z = self.masked_attn_norm(x + self.masked_attn_dropout(
                self.masked_attn(x, x, x, mask=forward_mask)
            ))
            # |z| = (batch_size, m, hidden_size)

        # |KV| = (batch_size, n, hidden_size)
        # |mask| = (batch_size, m, n)
        z = self.attn_norm(z + self.attn_dropout(self.attn(Q=z, K=KV, V=KV, mask=mask)))
        # |z| = (batch_size, m, hidden_size)

        z = self.fc_norm(z + self.fc_dropout(self.fc(z)))
        # |z| = (batch_size, m, hidden_size)

        return (
            z,
            KV,
            mask,
            prev,
        )


class MySequential(nn.Sequential):

    def forward(self, *x):
        for module in self._modules.values():
            x = module(*x)

        return x


class Transformer(nn.Module):
    
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_splits,
        n_enc_blocks=6,
        n_dec_blocks=6,
        dropout_p=.1,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_splits = n_splits
        self.n_enc_blocks = n_enc_blocks
        self.n_dec_blocks = n_dec_blocks
        self.dropout_p = dropout_p

        super().__init__()

        self.emb_enc = nn.Embedding(input_size, hidden_size)
        self.emb_dec = nn.Embedding(output_size, hidden_size)

        self.encoder = MySequential(
            *[EncoderBlock(hidden_size, n_splits, dropout_p) for _ in range(n_enc_blocks)]
        )
        self.decoder = MySequential(
            *[DecoderBlock(hidden_size, n_splits, dropout_p) for _ in range(n_dec_blocks)],
        )
        self.generator = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def _generate_mask(self, x, length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples, 
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat([x.new_ones(1, l).zero_(),
                                    x.new_ones(1, (max_length - l))
                                    ], dim=-1)]
            else:
                # If the length of the sample equals to maximum length among samples, 
                # set every value in mask to be 0.
                mask += [x.new_ones(1, l).zero_()]

        mask = torch.cat(mask, dim=0).byte()
        # |mask| = (batch_size, max_length)

        return mask

    def forward(self, x, y):
        # |x[0]| = (batch_size, n)
        # |y| = (batch_size, m)

        mask = self._generate_mask(x[0], x[1])
        # |mask| = (batch_size, n) 
        x = x[0]

        mask_enc = torch.stack([mask for _ in range(x.size(1))], dim=1)
        mask_dec = torch.stack([mask for _ in range(y.size(1))], dim=1)
        # |mask_enc| = (batch_size, n, n)
        # |mask_dec| = (batch_size, m, n)

        z = self.emb_enc(x)
        z, _ = self.encoder(z, mask_enc)        
        # |z| = (batch_size, n, hidden_size)

        h = self.emb_dec(y)
        h, _, _, _ = self.decoder(h, z, mask_dec, None)
        # |h| = (batch_size, m, hidden_size)

        y_hat = self.softmax(self.generator(h))
        # |y_hat| = (batch_size, m, output_size)

        return y_hat

    def search(self, x, is_greedy=True, max_length=255):
        # |x[0]| = (batch_size, n)
        batch_size = x[0].size(0)

        mask = self._generate_mask(x[0], x[1])
        # |mask| = (batch_size, n) 
        x = x[0]

        mask_enc = torch.stack([mask for _ in range(x.size(1))], dim=1)
        mask_dec = mask.unsqueeze(1)
        # |mask_enc| = (batch_size, n, n)
        # |mask_dec| = (batch_size, 1, n)

        z = self.emb_enc(x)
        z, _ = self.encoder(z, mask_enc)
        # |z| = (batch_size, n, hidden_size)

        # Fill a vector, which has 'batch_size' dimension, with BOS value.
        y_t_1 = x.new(batch_size, 1).zero_() + data_loader.BOS
        # |y_t_1| = (batch_size, 1)
        is_undone = x.new_ones(batch_size, 1).float()

        prevs, y_hats, indice = [[] for _ in range(len(self.decoder._modules) + 1)], [], []
        # Repeat a loop while sum of 'is_undone' flag is bigger than 0, or current time-step is smaller than maximum length.
        while is_undone.sum() > 0 and len(indice) < max_length:
            # Unlike training procedure, take the last time-step's output during the inference.
            h_t = self.emb_dec(y_t_1)
            # |h_t| = (batch_size, 1, hidden_size))
            prevs[0] += [h_t]

            for i, block in enumerate(self.decoder._modules.values()):
                prev = torch.cat(prevs[i], dim=1)
                # |prev| = (batch_size, m, hidden_size)

                h_t, _, _, _ = block(h_t, z, mask_dec, prev)
                # |h_t| = (batch_size, 1, hidden_size)

                prevs[i + 1] += [h_t]

            y_hat_t = self.softmax(self.generator(h_t))
            # |y_hat_t| = (batch_size, 1, output_size)

            y_hats += [y_hat_t]
            if is_greedy:
                y_t_1 = torch.topk(y_hat_t, 1, dim=-1)[1].squeeze(-1)
            else:
                # Take a random sampling based on the multinoulli distribution.
                y_t_1 = torch.multinomial(y_hat_t.exp().view(x.size(0), -1), 1)
            # Put PAD if the sample is done.
            y_t_1 = y_t_1.masked_fill_((1. - is_undone).byte(), data_loader.PAD)
            is_undone = is_undone * torch.ne(y_t_1, data_loader.EOS).float()
            # |y| = (batch_size, 1)
            # |is_undone| = (batch_size, 1)
            indice += [y_t_1]

        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=-1)
        # |y_hats| = (batch_size, m, output_size)
        # |indice| = (batch_size, m)

        return y_hats, indice
