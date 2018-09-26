import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.utils as torch_utils

import utils

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class Trainer():

    def __init__(self, model, crit, init_lr=1.):
        self.model = model
        self.crit = crit

        super().__init__()

        self.best = {'epoch': 0,
                     'lr': init_lr
                     }

    def get_best_model(self):
        self.model.load_state_dict(self.best['model'])

        return self.model

    def get_loss(self, y_hat, y, crit=None):
        # |y_hat| = (batch_size, length, output_size)
        # |y| = (batch_size, length)
        crit = self.crit if crit is None else crit
        loss = crit(y_hat.contiguous().view(-1, y_hat.size(-1)), y.contiguous().view(-1))

        return loss

    def train_epoch(self,
                    train,
                    optimizer,
                    batch_size=32,
                    max_grad_norm=5,
                    verbose=VERBOSE_SILENT
                    ):
        '''
        Train an epoch with given train iterator and optimizer.
        '''
        total_loss, total_word_count, total_param_norm, total_grad_norm = 0, 0, 0, 0
        avg_loss, avg_param_norm, avg_grad_norm = 0, 0, 0
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
            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)
            
            # You have to reset the gradients of all model parameters before to take another step in gradient descent.
            optimizer.zero_grad()

            # Take feed-forward
            # Similar as before, the input of decoder does not have EOS token.
            # Thus, remove EOS token for decoder input.
            y_hat = self.model(x, mini_batch.tgt[0][:, :-1])
            # |y_hat| = (batch_size, length, output_size)

            # Calcuate loss and gradients with back-propagation.
            loss = self.get_loss(y_hat, y)
            loss.backward()
            
            # Simple math to show stats.
            total_loss += loss
            total_word_count += int(mini_batch.tgt[1].sum())
            total_param_norm += utils.get_parameter_norm(self.model.parameters())
            total_grad_norm += utils.get_grad_norm(self.model.parameters())

            avg_loss = total_loss / total_word_count
            avg_param_norm = total_param_norm / (idx + 1)
            avg_grad_norm = total_grad_norm / (idx + 1)

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.set_postfix_str('|param|=%.2f |g_param|=%.2f loss=%.4e PPL=%.2f' % (avg_param_norm,
                                                                                                  avg_grad_norm,
                                                                                                  avg_loss,
                                                                                                  avg_loss.exp()
                                                                                                  ))

            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(self.model.parameters(),
                                        max_grad_norm
                                        )
            # Take a step of gradient descent.
            optimizer.step()

            sample_cnt += mini_batch.tgt[0].size(0)
            if sample_cnt >= len(train.dataset.examples):
                break
            
        if verbose is VERBOSE_BATCH_WISE:
            progress_bar.close()

        return avg_loss, avg_param_norm, avg_grad_norm

    def train(self, train, valid, config):
        '''
        Train with given train and valid iterator until n_epochs.
        If early_stop is set,
        early stopping will be executed if the requirement is satisfied.
        '''
        current_lr = self.best['lr']
        lowest_loss = float('Inf')
        lowest_after = 0

        progress_bar = tqdm(range(self.best['epoch'], config.n_epochs),
                            desc='Training: ',
                            unit='epoch'
                            ) if config.verbose is VERBOSE_EPOCH_WISE else range(self.best['epoch'],
                                                                                 config.n_epochs
                                                                                 )

        for idx in progress_bar:  # Iterate from 1 to n_epochs
            if config.adam:
                optimizer = optim.Adam(self.model.parameters(), lr=current_lr)
            else:
                optimizer = optim.SGD(self.model.parameters(), lr=current_lr)

            if config.verbose > VERBOSE_EPOCH_WISE:
                print('epoch: %d/%d\tmin_valid_loss=%.4e' % (idx + 1,
                                                             len(progress_bar),
                                                             lowest_loss
                                                             ))
            avg_train_loss, avg_param_norm, avg_grad_norm = self.train_epoch(train,
                                                                             optimizer,
                                                                             batch_size=config.batch_size,
                                                                             verbose=config.verbose
                                                                             )
            avg_valid_loss = self.validate(valid,
                                           verbose=config.verbose
                                           )

            # Print train status with different verbosity.
            if config.verbose is VERBOSE_EPOCH_WISE:
                progress_bar.set_postfix_str('|param|=%.2f |g_param|=%.2f train_loss=%.4e valid_loss=%.4e min_valid_loss=%.4e' % (float(avg_param_norm),
                                                                                                                                  float(avg_grad_norm),
                                                                                                                                  float(avg_train_loss),
                                                                                                                                  float(avg_valid_loss),
                                                                                                                                  float(lowest_loss)
                                                                                                                                  ))

            if avg_valid_loss < lowest_loss:
                # Update if there is an improvement.
                lowest_loss = avg_valid_loss
                lowest_after = 0

                self.best = {'model': self.model.state_dict(),
                             'optim': optimizer,
                             'epoch': idx,
                             'lowest_loss': lowest_loss,
                             'lr': current_lr
                             }
            else:
                lowest_after += 1

                if lowest_after >= config.early_stop and config.early_stop > 0:
                    break
        if config.verbose is VERBOSE_EPOCH_WISE:
            progress_bar.close()

    def validate(self,
                 valid,
                 crit=None,
                 batch_size=256,
                 verbose=VERBOSE_SILENT
                 ):
        '''
        Validate a model with given valid iterator.
        '''
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
                x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
                # |x| = (batch_size, length)
                # |y| = (batch_size, length)
                y_hat = self.model(x, mini_batch.tgt[0][:, :-1])
                # |y_hat| = (batch_size, n_classes)
                
                loss = self.get_loss(y_hat, y, crit)

                total_loss += float(loss)
                total_word_count += int(mini_batch.tgt[1].sum())
                avg_loss = total_loss / total_word_count

                sample_cnt += mini_batch.tgt[0].size(0)

                if verbose is VERBOSE_BATCH_WISE:
                    progress_bar.set_postfix_str('valid_loss=%.4e PPL=%.2f' % (avg_loss, avg_loss.exp()))

                if sample_cnt >= len(valid.dataset.examples):
                    break
            self.model.train()

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.close()

            return avg_loss
