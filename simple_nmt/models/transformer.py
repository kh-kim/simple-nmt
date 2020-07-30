import torch
import torch.nn as nn

import simple_nmt.data_loader as data_loader
from simple_nmt.search import SingleBeamSearchSpace


class Attention(nn.Module):

    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, dk=64):
        # |Q| = (batch_size, m, hidden_size)
        # |K| = |V| = (batch_size, n, hidden_size)
        # |mask| = (batch_size, m, n)

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

        # Note that we don't have to declare each linear layer, separately.
        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

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

        # By concatenating splited linear transformed results,
        # we can remove sequential operations,
        # like mini-batch parallel operations.
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
            dk=self.hidden_size // self.n_splits,
        )
        # |c| = (batch_size * n_splits, m, hidden_size / n_splits)

        # We need to restore temporal mini-batchfied multi-head attention results.
        c = c.split(Q.size(0), dim=0)
        # |c_i| = (batch_size, m, hidden_size / n_splits)
        c = self.linear(torch.cat(c, dim=-1))
        # |c| = (batch_size, m, hidden_size)

        return c


class EncoderBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        n_splits,
        dropout_p=.1,
        use_leaky_relu=False,
    ):
        super().__init__()

        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        # |x| = (batch_size, n, hidden_size)
        # |mask| = (batch_size, n, n)

        z = self.attn_norm(x + self.attn_dropout(self.attn(Q=x,
                                                           K=x,
                                                           V=x,
                                                           mask=mask)))
        z = self.fc_norm(z + self.fc_dropout(self.fc(z)))
        # |z| = (batch_size, n, hidden_size)

        return z, mask


class DecoderBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        n_splits,
        dropout_p=.1,
        use_leaky_relu=False,
    ):
        super().__init__()

        self.masked_attn = MultiHead(hidden_size, n_splits)
        self.masked_attn_norm = nn.LayerNorm(hidden_size)
        self.masked_attn_dropout = nn.Dropout(dropout_p)

        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, key_and_value, mask, prev):
        # In case of inference, we don't have to repeat same feed-forward operations.
        # Thus, we save previous feed-forward results, including intermediate results.
        if prev is not None:
            # |x| = (batch_size, m=1, hidden_size)
            # |prev| = (batch_size, m', hidden_size)

            z = self.masked_attn_norm(x + self.masked_attn_dropout(
                self.masked_attn(x, prev, prev, mask=None)
            ))
            # |z| = (batch_size, 1, hidden_size)
        else:
            # |x| = (batch_size, m, hidden_size)
            batch_size = x.size(0)
            m = x.size(1)

            fwd_mask = torch.triu(x.new_ones((m, m)), diagonal=1).bool()
            # |fwd_mask| = (m, m)
            fwd_mask = fwd_mask.unsqueeze(0).expand(batch_size, *fwd_mask.size())
            # |fwd_mask| = (batch_size, m, m)

            z = self.masked_attn_norm(x + self.masked_attn_dropout(
                self.masked_attn(x, x, x, mask=fwd_mask)
            ))
            # |z| = (batch_size, m, hidden_size)

        # |key_and_value| = (batch_size, n, hidden_size)
        # |mask| = (batch_size, m, n)
        z = self.attn_norm(z + self.attn_dropout(self.attn(Q=z,
                                                           K=key_and_value,
                                                           V=key_and_value,
                                                           mask=mask)))
        # |z| = (batch_size, m, hidden_size)

        z = self.fc_norm(z + self.fc_dropout(self.fc(z)))
        # |z| = (batch_size, m, hidden_size)

        return z, key_and_value, mask, prev


class MySequential(nn.Sequential):

    def forward(self, *x):
        # nn.Sequential class does not provide multiple input arguments and returns.
        # Thus, we need to define new class to solve this issue.
        # Note that each block has same function interface.

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
        use_leaky_relu=False,
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
        self.emb_dropout = nn.Dropout(dropout_p)

        self.encoder = MySequential(
            *[EncoderBlock(
                hidden_size,
                n_splits,
                dropout_p,
                use_leaky_relu,
              ) for _ in range(n_enc_blocks)],
        )
        self.decoder = MySequential(
            *[DecoderBlock(
                hidden_size,
                n_splits,
                dropout_p,
                use_leaky_relu,
              ) for _ in range(n_dec_blocks)],
        )
        self.generator = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def _position_encoding(self, x, init_pos=0):
        # |x| = (batch_size, n, hidden_size)
        length, hidden_size = x.size(1), x.size(-1)

        enc = x.new_zeros(x.shape[1:])
        # |enc| = (n, hidden_size)
        pos = init_pos + torch.arange(0, length, device=x.device).unsqueeze(-1)
        dim = (
            1e+4**torch.arange(0, hidden_size // 2, device=x.device).div(float(hidden_size))
        ).unsqueeze(0)
        # |pos| = (n, 1)
        # |dim| = (1, hidden_size // 2)

        assert enc[:, 0::2].size() == (pos / dim).size()
        assert enc[:, 1::2].size() == (pos / dim).size()

        pos = pos.float()
        dim = dim.float()

        enc[:, 0::2] = torch.sin(pos / dim)
        enc[:, 1::2] = torch.cos(pos / dim)

        x = x + enc

        return x

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
                # If length of sample equals to maximum length among samples,
                # set every value in mask to be 0.
                mask += [x.new_ones(1, l).zero_()]

        mask = torch.cat(mask, dim=0).bool()
        # |mask| = (batch_size, max_length)

        return mask

    def forward(self, x, y):
        # |x[0]| = (batch_size, n)
        # |y| = (batch_size, m)

        mask = self._generate_mask(x[0], x[1])
        # |mask| = (batch_size, n)
        x = x[0]

        mask_enc = mask.unsqueeze(1).expand(mask.size(0), x.size(1), mask.size(-1))
        mask_dec = mask.unsqueeze(1).expand(mask.size(0), y.size(1), mask.size(-1))
        # |mask_enc| = (batch_size, n, n)
        # |mask_dec| = (batch_size, m, n)

        z = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
        z, _ = self.encoder(z, mask_enc)
        # |z| = (batch_size, n, hidden_size)

        h = self.emb_dropout(self._position_encoding(self.emb_dec(y)))
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

        mask_enc = mask.unsqueeze(1).expand(mask.size(0), x.size(1), mask.size(-1))
        mask_dec = mask.unsqueeze(1)
        # |mask_enc| = (batch_size, n, n)
        # |mask_dec| = (batch_size, 1, n)

        z = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
        z, _ = self.encoder(z, mask_enc)
        # |z| = (batch_size, n, hidden_size)

        # Fill a vector, which has 'batch_size' dimension, with BOS value.
        y_t_1 = x.new(batch_size, 1).zero_() + data_loader.BOS
        # |y_t_1| = (batch_size, 1)
        is_undone = x.new_ones(batch_size, 1).float()

        prevs = [None for _ in range(len(self.decoder._modules) + 1)]
        y_hats, indice = [], []
        # Repeat a loop while sum of 'is_undone' flag is bigger than 0,
        # or current time-step is smaller than maximum length.
        while is_undone.sum() > 0 and len(indice) < max_length:
            # Unlike training procedure,
            # take the last time-step's output during the inference.
            h_t = self.emb_dropout(
                self._position_encoding(self.emb_dec(y_t_1), init_pos=len(indice))
            )
            # |h_t| = (batch_size, 1, hidden_size))
            if prevs[0] is None:
                prevs[0] = h_t
            else:
                prevs[0] = torch.cat([prevs[0], h_t], dim=1)

            for layer_idx, block in enumerate(self.decoder._modules.values()):
                prev = prevs[layer_idx]
                # |prev| = (batch_size, m, hidden_size)

                h_t, _, _, _ = block(h_t, z, mask_dec, prev)
                # |h_t| = (batch_size, 1, hidden_size)

                if prevs[layer_idx + 1] is None:
                    prevs[layer_idx + 1] = h_t
                else:
                    prevs[layer_idx + 1] = torch.cat([prevs[layer_idx + 1], h_t], dim=1)

            y_hat_t = self.softmax(self.generator(h_t))
            # |y_hat_t| = (batch_size, 1, output_size)

            y_hats += [y_hat_t]
            if is_greedy:
                y_t_1 = torch.topk(y_hat_t, 1, dim=-1)[1].squeeze(-1)
            else:
                # Take a random sampling based on the multinoulli distribution.
                y_t_1 = torch.multinomial(y_hat_t.exp().view(x.size(0), -1), 1)
            # Put PAD if the sample is done.
            y_t_1 = y_t_1.masked_fill_(
                (1. - is_undone).bool(),
                data_loader.PAD,
            )
            is_undone = is_undone * torch.ne(y_t_1, data_loader.EOS).float()
            # |y| = (batch_size, 1)
            # |is_undone| = (batch_size, 1)
            indice += [y_t_1]

        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=-1)
        # |y_hats| = (batch_size, m, output_size)
        # |indice| = (batch_size, m)

        return y_hats, indice

    def batch_beam_search(
        self,
        x,
        beam_size=5,
        max_length=255,
        n_best=1,
        length_penalty=.2
    ):
        # |x[0]| = (batch_size, n)
        batch_size = x[0].size(0)

        mask = self._generate_mask(x[0], x[1])
        # |mask| = (batch_size, n)
        x = x[0]

        mask_enc = mask.unsqueeze(1).expand(mask.size(0), x.size(1), mask.size(-1))
        mask_dec = mask.unsqueeze(1)
        # |mask_enc| = (batch_size, n, n)
        # |mask_dec| = (batch_size, 1, n)

        z = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
        z, _ = self.encoder(z, mask_enc)
        # |z| = (batch_size, n, hidden_size)

        spaces = [SingleBeamSearchSpace(
            z.device,
            [('prev_state_%d' % j, None, 0) for j in range(len(self.decoder._modules) + 1)],
            beam_size=beam_size,
            max_length=max_length,
        ) for i in range(batch_size)]
        done_cnt = [space.is_done() for space in spaces]

        length = 0
        while sum(done_cnt) < batch_size and length <= max_length:
            fab_input, fab_z, fab_mask = [], [], []
            fab_prevs = [[] for _ in range(len(self.decoder._modules) + 1)]

            for i, space in enumerate(spaces):
                if space.is_done() == 0:
                    tmp = space.get_batch()

                    y_hat_ = tmp[0]
                    tmp = tmp[1:]

                    fab_input += [y_hat_]
                    for j, prev_ in enumerate(tmp):
                        if prev_ is not None:
                            fab_prevs[j] += [prev_]
                        else:
                            fab_prevs[j] = None

                    fab_z += [z[i].unsqueeze(0)] * beam_size
                    fab_mask += [mask_dec[i].unsqueeze(0)] * beam_size

            fab_input = torch.cat(fab_input, dim=0)
            for i, fab_prev in enumerate(fab_prevs):
                if fab_prev is not None:
                    fab_prevs[i] = torch.cat(fab_prev, dim=0)
            fab_z = torch.cat(fab_z, dim=0)
            fab_mask = torch.cat(fab_mask, dim=0)
            # |fab_input| = (current_batch_size, 1,)
            # |fab_prevs[i]| = (current_batch_size, length, hidden_size)
            # |fab_z| = (current_batch_size, n, hidden_size)
            # |fab_mask| = (current_batch_size, 1, n)

            # Unlike training procedure,
            # take the last time-step's output during the inference.
            h_t = self.emb_dropout(
                self._position_encoding(self.emb_dec(fab_input), init_pos=length)
            )
            # |h_t| = (current_batch_size, 1, hidden_size)
            if fab_prevs[0] is None:
                fab_prevs[0] = h_t
            else:
                fab_prevs[0] = torch.cat([fab_prevs[0], h_t], dim=1)

            for layer_idx, block in enumerate(self.decoder._modules.values()):
                prev = fab_prevs[layer_idx]
                # |prev| = (current_batch_size, m, hidden_size)

                h_t, _, _, _ = block(h_t, fab_z, fab_mask, prev)
                # |h_t| = (current_batch_size, 1, hidden_size)

                if fab_prevs[layer_idx + 1] is None:
                    fab_prevs[layer_idx + 1] = h_t
                else:
                    fab_prevs[layer_idx + 1] = torch.cat(
                        [fab_prevs[layer_idx + 1], h_t],
                        dim=1,
                    )

            y_hat_t = self.softmax(self.generator(h_t))
            # |y_hat_t| = (batch_size, 1, output_size)

            # |fab_prevs[i][from_index:to_index]| = (beam_size, length, hidden_size)
            cnt = 0
            for space in spaces:
                if space.is_done() == 0:
                    from_index = cnt * beam_size
                    to_index = from_index + beam_size

                    space.collect_result(
                        y_hat_t[from_index:to_index],
                        [
                            (
                                'prev_state_%d' % i,
                                fab_prevs[i][from_index:to_index],
                            ) for i in range(len(self.decoder._modules) + 1)
                        ],
                    )

                    cnt += 1

            done_cnt = [space.is_done() for space in spaces]
            length += 1

        batch_sentences = []
        batch_probs = []

        for i, space in enumerate(spaces):
            sentences, probs = space.get_n_best(n_best, length_penalty=length_penalty)

            batch_sentences += [sentences]
            batch_probs += [probs]

        return batch_sentences, batch_probs
