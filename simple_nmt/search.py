from operator import itemgetter

import torch
import torch.nn as nn

import simple_nmt.data_loader as data_loader

LENGTH_PENALTY = .2
MIN_LENGTH = 5


class SingleBeamSearchBoard():

    def __init__(
        self,
        device,
        prev_status_config,
        beam_size=5,
        max_length=255,
    ):
        self.beam_size = beam_size
        self.max_length = max_length

        # To put data to same device.
        self.device = device
        # Inferred word index for each time-step. For now, initialized with initial time-step.
        self.word_indice = [torch.LongTensor(beam_size).zero_().to(self.device) + data_loader.BOS]
        # Beam index for selected word index, at each time-step.
        self.beam_indice = [torch.LongTensor(beam_size).zero_().to(self.device) - 1]
        # Cumulative log-probability for each beam.
        self.cumulative_probs = [torch.FloatTensor([.0] + [-float('inf')] * (beam_size - 1)).to(self.device)]
        # 1 if it is done else 0
        self.masks = [torch.BoolTensor(beam_size).zero_().to(self.device)]

        # We don't need to remember every time-step of hidden states:
        #       prev_hidden, prev_cell, prev_h_t_tilde
        # What we need is remember just last one.

        self.prev_status = {}
        self.batch_dims = {}
        for prev_status_name, each_config in prev_status_config.items():
            init_status = each_config['init_status']
            batch_dim_index = each_config['batch_dim_index']
            if init_status is not None:
                self.prev_status[prev_status_name] = torch.cat([init_status] * beam_size,
                                                               dim=batch_dim_index)
            else:
                self.prev_status[prev_status_name] = None
            self.batch_dims[prev_status_name] = batch_dim_index

        self.current_time_step = 0
        self.done_cnt = 0

    def get_length_penalty(
        self,
        length,
        alpha=LENGTH_PENALTY,
        min_length=MIN_LENGTH,
    ):
        # Calculate length-penalty,
        # because shorter sentence usually have bigger probability.
        # In fact, we represent this as log-probability, which is negative value.
        # Thus, we need to multiply bigger penalty for shorter one.
        p = ((min_length + 1) / (min_length + length))**alpha

        return p

    def is_done(self):
        # Return 1, if we had EOS more than 'beam_size'-times.
        if self.done_cnt >= self.beam_size:
            return 1
        return 0

    def get_batch(self):
        y_hat = self.word_indice[-1].unsqueeze(-1)
        # |y_hat| = (beam_size, 1)
        # if model != transformer:
        #     |hidden| = |cell| = (n_layers, beam_size, hidden_size)
        #     |h_t_tilde| = (beam_size, 1, hidden_size) or None
        # else:
        #     |prev_state_i| = (beam_size, length, hidden_size),
        #     where i is an index of layer.
        return y_hat, self.prev_status

    #@profile
    def collect_result(self, y_hat, prev_status):
        # |y_hat| = (beam_size, 1, output_size)
        # prev_status is a dict, which has following keys:
        # if model != transformer:
        #     |hidden| = |cell| = (n_layers, beam_size, hidden_size)
        #     |h_t_tilde| = (beam_size, 1, hidden_size)
        # else:
        #     |prev_state_i| = (beam_size, length, hidden_size),
        #     where i is an index of layer.
        output_size = y_hat.size(-1)

        self.current_time_step += 1

        # Calculate cumulative log-probability.
        # First, fill -inf value to last cumulative probability, if the beam is already finished.
        # Second, expand -inf filled cumulative probability to fit to 'y_hat'.
        # (beam_size) --> (beam_size, 1, 1) --> (beam_size, 1, output_size)
        # Third, add expanded cumulative probability to 'y_hat'
        cumulative_prob = self.cumulative_probs[-1].masked_fill_(self.masks[-1], -float('inf'))
        cumulative_prob = y_hat + cumulative_prob.view(-1, 1, 1).expand(self.beam_size, 1, output_size)
        # |cumulative_prob| = (beam_size, 1, output_size)

        # Now, we have new top log-probability and its index.
        # We picked top index as many as 'beam_size'.
        # Be aware that we picked top-k from whole batch through 'view(-1)'.

        # Following lines are using torch.topk, which is slower than torch.sort.
        # top_log_prob, top_indice = torch.topk(
        #     cumulative_prob.view(-1), # (beam_size * output_size,)
        #     self.beam_size,
        #     dim=-1,
        # )

        # Following lines are using torch.sort, instead of using torch.topk.
        top_log_prob, top_indice = cumulative_prob.view(-1).sort(descending=True)
        top_log_prob, top_indice = top_log_prob[:self.beam_size], top_indice[:self.beam_size]

        # |top_log_prob| = (beam_size,)
        # |top_indice| = (beam_size,)

        # Because we picked from whole batch, original word index should be calculated again.
        self.word_indice += [top_indice.fmod(output_size)]
        # Also, we can get an index of beam, which has top-k log-probability search result.
        self.beam_indice += [top_indice.div(float(output_size)).long()]

        # Add results to history boards.
        self.cumulative_probs += [top_log_prob]
        self.masks += [torch.eq(self.word_indice[-1], data_loader.EOS)] # Set finish mask if we got EOS.
        # Calculate a number of finished beams.
        self.done_cnt += self.masks[-1].float().sum()

        # In beam search procedure, we only need to memorize latest status.
        # For seq2seq, it would be lastest hidden and cell state, and h_t_tilde.
        # The problem is hidden(or cell) state and h_t_tilde has different dimension order.
        # In other words, a dimension for batch index is different.
        # Therefore self.batch_dims stores the dimension index for batch index.
        # For transformer, lastest status is each layer's decoder output from the biginning.
        # Unlike seq2seq, transformer has to memorize every previous output for attention operation.
        for prev_status_name, prev_status in prev_status.items():
            self.prev_status[prev_status_name] = torch.index_select(
                prev_status,
                dim=self.batch_dims[prev_status_name],
                index=self.beam_indice[-1]
            ).contiguous()

    def get_n_best(self, n=1, length_penalty=.2):
        sentences, probs, founds = [], [], []

        for t in range(len(self.word_indice)):  # for each time-step,
            for b in range(self.beam_size):  # for each beam,
                if self.masks[t][b] == 1:  # if we had EOS on this time-step and beam,
                    # Take a record of penaltified log-proability.
                    probs += [self.cumulative_probs[t][b] * self.get_length_penalty(t, alpha=length_penalty)]
                    founds += [(t, b)]

        # Also, collect log-probability from last time-step, for the case of EOS is not shown.
        for b in range(self.beam_size):
            if self.cumulative_probs[-1][b] != -float('inf'): # If this beam does not have EOS,
                if not (len(self.cumulative_probs) - 1, b) in founds:
                    probs += [self.cumulative_probs[-1][b] * self.get_length_penalty(len(self.cumulative_probs),
                                                                                     alpha=length_penalty)]
                    founds += [(t, b)]

        # Sort and take n-best.
        sorted_founds_with_probs = sorted(
            zip(founds, probs),
            key=itemgetter(1),
            reverse=True,
        )[:n]
        probs = []

        for (end_index, b), prob in sorted_founds_with_probs:
            sentence = []

            # Trace from the end.
            for t in range(end_index, 0, -1):
                sentence = [self.word_indice[t][b]] + sentence
                b = self.beam_indice[t][b]

            sentences += [sentence]
            probs += [prob]

        return sentences, probs
