from tqdm import tqdm
from math import exp

import torch
import torch.optim as optim
import torch.nn.utils as torch_utils

import utils

from simple_nmt.trainer import Trainer

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class LanguageModelTrainer(Trainer):

    def __init__(self, model, crit, config, is_src=True, **kwargs):
        self.is_src = is_src

        super().__init__(model=model, crit=crit, config=config, **kwargs)

        self.n_epochs = config.lm_n_epochs

    def train_epoch(self, train, optimizer, verbose=VERBOSE_BATCH_WISE):
        '''
        Train an epoch with given train iterator and optimizer.
        '''
        total_loss, total_word_count = 0, 0
        total_grad_norm = 0
        avg_loss, avg_grad_norm = 0, 0
        sample_cnt = 0

        progress_bar = tqdm(train,
                            desc='Training: ',
                            unit='batch'
                            ) if verbose is VERBOSE_BATCH_WISE else train
        # Iterate whole train-set.
        for idx, mini_batch in enumerate(progress_bar):
            # Raw target variable has both BOS and EOS token. 
            # The output of sequence-to-sequence does not have BOS token. 
            # Thus, remove BOS token for reference.
            x = mini_batch.src[0][:, :-1] if self.is_src else mini_batch.tgt[0][:, :-1]
            y = mini_batch.src[0][:, 1:] if self.is_src else mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length)

            # You have to reset the gradients of all model parameters before to take another step in gradient descent.
            optimizer.zero_grad()

            # Take feed-forward
            y_hat = self.model(x)
            loss = self._get_loss(y_hat, y).sum()
            loss.div(y.size(0)).backward()

            # Simple math to show stats.
            # Don't forget to detach final variables.
            total_loss += float(loss)
            total_word_count += int(mini_batch.src[1].sum() if self.is_src else mini_batch.tgt[1].sum())
            param_norm = float(utils.get_parameter_norm(self.model.parameters()))
            total_grad_norm += float(utils.get_grad_norm(self.model.parameters()))

            avg_loss = total_loss / total_word_count
            avg_grad_norm = total_grad_norm / (idx + 1)

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.set_postfix_str('|param|=%.2f |g_param|=%.2f loss=%.4e PPL=%.2f' % (param_norm,
                                                                                                 avg_grad_norm,
                                                                                                 avg_loss,
                                                                                                 exp(avg_loss)
                                                                                                 ))

            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(self.model.parameters(),
                                        self.config.max_grad_norm
                                        )
            # Take a step of gradient descent.
            optimizer.step()

            sample_cnt += x.size(0)
            if idx >= len(progress_bar):
                break

        if verbose is VERBOSE_BATCH_WISE:
            progress_bar.close()

        return avg_loss, param_norm, avg_grad_norm


    def train(self, train, valid, verbose=VERBOSE_EPOCH_WISE):
        current_lr = self.best['current_lr']
        best_loss = float('Inf') * (1 if self.lower_is_better else -1)
        lowest_after = 0

        progress_bar = tqdm(range(self.best['epoch'], self.n_epochs),
                            desc='Training: ',
                            unit='epoch'
                            ) if verbose is VERBOSE_EPOCH_WISE else range(self.best['epoch'],
                                                                          self.n_epochs
                                                                          )

        if self.best['epoch'] > 0:
            avg_valid_loss = self.validate(valid, verbose=verbose)

        optimizer = optim.Adam(self.model.parameters())
        for idx in progress_bar:  # Iterate from 1 to n_epochs            
            if verbose > VERBOSE_EPOCH_WISE:
                print('epoch: %d/%d\tmin_valid_loss=%.4e' % (idx + 1,
                                                             self.n_epochs,
                                                             best_loss
                                                             ))
            avg_train_loss, avg_param_norm, avg_grad_norm = self.train_epoch(train,
                                                                             optimizer,
                                                                             verbose=verbose
                                                                             )
            avg_valid_loss = self.validate(valid, verbose=verbose)

            # Print train status with different verbosity.
            if verbose is VERBOSE_EPOCH_WISE:
                progress_bar.set_postfix_str('|param|=%.2f |g_param|=%.2f train_loss=%.4e valid_loss=%.4e min_valid_loss=%.4e' % (avg_param_norm,
                                                                                                                                  avg_grad_norm,
                                                                                                                                  avg_train_loss,
                                                                                                                                  avg_valid_loss,
                                                                                                                                  best_loss
                                                                                                                                  ))

            if (self.lower_is_better and avg_valid_loss < best_loss) or \
               (not self.lower_is_better and avg_valid_loss > best_loss):
                # Update if there is an improvement.
                best_loss = avg_valid_loss
                lowest_after = 0

                self.best['model'] = self.model.state_dict()
                self.best['optim'] = optimizer
                self.best['epoch'] = idx + 1
                self.best['current_lr'] = current_lr
            else:
                lowest_after += 1

                if lowest_after >= self.config.early_stop and \
                   self.config.early_stop > 0:
                    break

            # Altough there is an improvement in last epoch, we need to decay the learning-rate if it meets the requirements.
            if ((lowest_after > 0) or (idx + 1 >= self.config.lr_decay_start_at)) and (idx + 1 <= self.config.n_epochs):
                current_lr = max(self.config.min_lr,
                                 current_lr * self.config.lr_decay_rate
                                 )

        if verbose is VERBOSE_EPOCH_WISE:
            progress_bar.close()

    def validate(self, valid, crit=None, verbose=VERBOSE_BATCH_WISE):
        # We don't need to back-prop for these operations.
        with torch.no_grad():
            sample_cnt = 0
            total_loss, total_word_count = 0, 0

            progress_bar = tqdm(valid,
                                desc='Validation: ',
                                unit='batch'
                                ) if verbose is VERBOSE_BATCH_WISE else valid

            self.model.eval()
            # Iterate for whole valid-set.
            for idx, mini_batch in enumerate(progress_bar):
                x = mini_batch.src[0][:, :-1] if self.is_src else mini_batch.tgt[0][:, :-1]
                y = mini_batch.src[0][:, 1:] if self.is_src else mini_batch.tgt[0][:, 1:]
                # |x| = (batch_size, length)
                # |y| = (batch_size, length)
                y_hat = self.model(x)
                # |y_hat| = (batch_size, n_classes)

                loss = self._get_loss(y_hat, y, crit).sum()

                total_loss += float(loss)
                total_word_count += int(mini_batch.src[1].sum() if self.is_src else mini_batch.tgt[1].sum())
                avg_loss = total_loss / total_word_count

                sample_cnt += x.size(0)

                if verbose is VERBOSE_BATCH_WISE:
                    progress_bar.set_postfix_str('valid_loss=%.4e PPL=%.2f' % (avg_loss,
                                                                               exp(avg_loss)
                                                                               ))
                if idx >= len(progress_bar):
                    break

            self.model.train()

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.close()

            return avg_loss
