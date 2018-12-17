from tqdm import tqdm
from operator import itemgetter
from math import exp

import torch
import torch.optim as optim
import torch.nn.utils as torch_utils

import utils

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

# In order to avoid to use hard coding.
X2Y, Y2X = 0, 1


class DualTrainer():

    def __init__(self, models, crits, language_models, config, **kwargs):
        self.models = models
        self.crits = crits
        self.language_models = language_models
        self.config = config

        super().__init__()

        self.n_epochs = config.n_epochs
        self.best = {'epoch': 0,
                     'config': config,
                     **kwargs
                     }

    def save_training(self, fn):
        torch.save(self.best, fn)

    def get_best_model(self):
        self.models[X2Y].load_state_dict(self.best['models'][X2Y])
        self.models[Y2X].load_state_dict(self.best['models'][Y2X])

        self.language_models[X2Y].load_state_dict(self.best['lms'][X2Y])
        self.language_models[Y2X].load_state_dict(self.best['lms'][Y2X])

        return self.models

    def _reordering(self, x, y, l):
        # This method is one of important methods in this class.
        # Since encoder takes packed_sequence instance,
        # the samples in mini-batch must be sorted by lengths.
        # Thus, we need to re-order the samples in mini-batch, if src and tgt is reversed.
        # (Because originally src and tgt are sorted by the length of samples in src.)

        # sort by length.
        indice = l.topk(l.size(0))[1]

        # re-order based on the indice.
        x_ = x.index_select(dim=0, index=indice).contiguous()
        y_ = y.index_select(dim=0, index=indice).contiguous()
        l_ = l.index_select(dim=0, index=indice).contiguous()

        # generate information to restore the re-ordering.
        restore_indice = (-indice).topk(l.size(0))[1]

        return x_, y_, l_, restore_indice

    def _get_loss(self, x, y, x_hat, y_hat, x_lm=None, y_lm=None, lagrange=1e-3):
        # |x| = (batch_size, length0)
        # |y| = (batch_size, length1)
        # |x_hat| = (batch_size, length0, output_size0)
        # |y_hat| = (batch_size, length1, output_size1)
        # |x_lm| = |x_hat|
        # |y_lm| = |y_hat|

        losses = []
        losses += [self.crits[X2Y](y_hat.contiguous().view(-1, y_hat.size(-1)),
                                   y.contiguous().view(-1)
                                   )]
        losses += [self.crits[Y2X](x_hat.contiguous().view(-1, x_hat.size(-1)),
                                   x.contiguous().view(-1)
                                   )]
        # |losses[X2Y]| = (batch_size * length1)
        # |losses[Y2X]| = (batch_size * length0)

        losses[X2Y] = losses[X2Y].view(y.size(0), -1).sum(dim=-1)
        losses[Y2X] = losses[Y2X].view(x.size(0), -1).sum(dim=-1)
        # |losses[X2Y]| = (batch_size)
        # |losses[Y2X]| = (batch_size)

        if x_lm is not None and y_lm is not None:
            lm_losses = []
            lm_losses += [self.crits[X2Y](y_lm.contiguous().view(-1, y_lm.size(-1)),
                                          y.contiguous().view(-1)
                                          )]
            lm_losses += [self.crits[Y2X](x_lm.contiguous().view(-1, x_lm.size(-1)),
                                          x.contiguous().view(-1)
                                          )]
            # |lm_losses[X2Y]| = (batch_size * length1)
            # |lm_losses[Y2X]| = (batch_size * length0)

            lm_losses[X2Y] = lm_losses[X2Y].view(y.size(0), -1).sum(dim=-1)
            lm_losses[Y2X] = lm_losses[Y2X].view(x.size(0), -1).sum(dim=-1)
            # |lm_losses[X2Y]| = (batch_size)
            # |lm_losses[Y2X]| = (batch_size)

            # just for information
            dual_loss = lagrange * ((-lm_losses[Y2X] + -losses[X2Y].detach()) - (-lm_losses[X2Y] + -losses[Y2X].detach()))**2

            # Note that 'detach()' is following the loss for another direction.
            dual_loss_x2y = lagrange * ((-lm_losses[Y2X] + -losses[X2Y]) - (-lm_losses[X2Y] + -losses[Y2X].detach()))**2
            dual_loss_y2x = lagrange * ((-lm_losses[Y2X] + -losses[X2Y].detach()) - (-lm_losses[X2Y] + -losses[Y2X]))**2

            losses[X2Y] += dual_loss_x2y
            losses[Y2X] += dual_loss_y2x

        if x_lm is not None and y_lm is not None:
            return losses[X2Y].sum(), losses[Y2X].sum(), dual_loss.sum()
        else:
            return losses[X2Y].sum(), losses[Y2X].sum(), None

    def train_epoch(self,
                    train,
                    optimizers,
                    no_regularization=True,
                    verbose=VERBOSE_BATCH_WISE
                    ):
        '''
        Train an epoch with given train iterator and optimizers.
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
            
            # You have to reset the gradients of all model parameters before to take another step in gradient descent.
            optimizers[X2Y].zero_grad()
            optimizers[Y2X].zero_grad()

            x_0, y_0 = (mini_batch.src[0][:, 1:-1],  # Remove BOS and EOS
                        mini_batch.src[1] - 2
                        ), mini_batch.tgt[0][:, :-1]
            # |x_0| = (batch_size, length0)
            # |y_0| = (batch_size, length1)
            y_hat = self.models[X2Y](x_0, y_0)
            # |y_hat| = (batch_size, length1, output_size1)
            with torch.no_grad():
                y_lm = self.language_models[X2Y](y_0)
                # |y_lm| = |y_hat|

            # Since encoder in seq2seq takes packed_sequence instance,
            # we need to re-sort if we use reversed src and tgt.
            x_0, y_0_0, y_0_1, restore_indice = self._reordering(mini_batch.src[0][:, :-1],
                                                                 mini_batch.tgt[0][:, 1:-1], # Remove BOS and EOS
                                                                 mini_batch.tgt[1] - 2
                                                                 )
            y_0 = (y_0_0, y_0_1)
            # |x_0| = (batch_size, length0)
            # |y_0| = (batch_size, length1)
            x_hat = self.models[Y2X](y_0, x_0).index_select(dim=0, index=restore_indice)
            # |x_hat| = (batch_size, length0, output_size0)

            with torch.no_grad():
                x_lm = self.language_models[Y2X](x_0)
                # |x_lm| = |x_hat|

            x, y = mini_batch.src[0][:, 1:], mini_batch.tgt[0][:, 1:]
            losses = self._get_loss(x,
                                    y,
                                    x_hat,
                                    y_hat,
                                    x_lm,
                                    y_lm,
                                    # According to the paper, DSL should be warm-started.
                                    # Thus, we turn-off the regularization at the beginning.
                                    lagrange=self.config.dsl_lambda if not no_regularization else .0
                                    )
            
            losses[X2Y].div(y.size(0)).backward()
            losses[Y2X].div(x.size(0)).backward()

            word_count = int((mini_batch.src[1].detach().sum()) + 
                             (mini_batch.tgt[1].detach().sum())
                             )
            loss = float(losses[X2Y].detach() + losses[Y2X].detach()) - float(losses[-1].detach() * 2)
            param_norm = float(utils.get_parameter_norm(self.models[X2Y].parameters()).detach() + 
                               utils.get_parameter_norm(self.models[Y2X].parameters()).detach()
                               )
            grad_norm = float(utils.get_grad_norm(self.models[X2Y].parameters()).detach() +
                              utils.get_grad_norm(self.models[Y2X].parameters()).detach()
                              )  

            total_loss += loss
            total_word_count += word_count
            total_grad_norm += grad_norm

            avg_loss = total_loss / total_word_count
            avg_grad_norm = total_grad_norm / (idx + 1)

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.set_postfix_str('|param|=%.2f |g_param|=%.2f loss=%.4e PPL=%.2f' % (param_norm,
                                                                                                 grad_norm,
                                                                                                 loss / word_count,
                                                                                                 exp(avg_loss)
                                                                                                 ))

            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(self.models[X2Y].parameters(),
                                        self.config.max_grad_norm
                                        )
            torch_utils.clip_grad_norm_(self.models[Y2X].parameters(),
                                        self.config.max_grad_norm
                                        )
            
            # Take a step of gradient descent.
            optimizers[X2Y].step()
            optimizers[Y2X].step()

            sample_cnt += mini_batch.tgt[0].size(0)

            if idx >= len(progress_bar) * self.config.train_ratio_per_epoch:
                break

        if verbose is VERBOSE_BATCH_WISE:
            progress_bar.close()

        return avg_loss, param_norm, avg_grad_norm

    def train(self, train, valid, verbose=VERBOSE_EPOCH_WISE):
        '''
        Train with given train and valid iterator until n_epochs.
        If early_stop is set,
        early stopping will be executed if the requirement is satisfied.
        '''
        best_loss = float('Inf')
        lowest_after = 0

        progress_bar = tqdm(range(self.best['epoch'], self.n_epochs),
                            desc='Training: ',
                            unit='epoch'
                            ) if verbose is VERBOSE_EPOCH_WISE else range(self.best['epoch'],
                                                                          self.n_epochs
                                                                          )

        optimizers = [optim.Adam(self.models[X2Y].parameters()),
                      optim.Adam(self.models[Y2X].parameters())]
        for idx in progress_bar:  # Iterate from 1 to n_epochs
            if verbose > VERBOSE_EPOCH_WISE:
                print('epoch: %d/%d\tmin_valid_loss=%.4e' % (idx + 1,
                                                             self.n_epochs,
                                                             best_loss
                                                             ))
            # Turn off the dual regularization term before the DSL start.
            if idx < self.config.n_epochs - self.config.dsl_n_epochs:
                no_regularization = True
            else:
                no_regularization = False
            print('apply duality term on loss: ', not no_regularization)
            avg_train_loss, avg_param_norm, avg_grad_norm = self.train_epoch(train,
                                                                             optimizers,
                                                                             no_regularization=True if idx < (self.config.n_epochs - self.config.dsl_n_epochs) else False,
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

            if avg_valid_loss < best_loss or idx >= (self.config.n_epochs - self.config.dsl_n_epochs):
                # Update if there is an improvement.
                best_loss = avg_valid_loss
                lowest_after = 0

                self.best['models'] = [self.models[X2Y].state_dict(),
                                       self.models[Y2X].state_dict()
                                       ]
                self.best['lms'] = [self.language_models[X2Y].state_dict(),
                                    self.language_models[Y2X].state_dict()
                                    ]
                self.best['optim'] = optimizers
                self.best['epoch'] = idx + 1

                # Set a filename for model of last epoch.
                # We need to put every information to filename, as much as possible.
                model_fn = self.config.model.split('.')
                model_fn = model_fn[:-1] + ['%02d' % (idx + 1),
                                            '%.2f-%.2f' % (avg_train_loss,
                                                           exp(avg_train_loss)
                                                           ),
                                            '%.2f-%.2f' % (avg_valid_loss,
                                                           exp(avg_valid_loss)
                                                           )
                                            ] + [model_fn[-1]]
                self.save_training('.'.join(model_fn))
            else:
                lowest_after += 1

                if lowest_after >= self.config.early_stop and \
                   self.config.early_stop > 0:
                    break
        if verbose is VERBOSE_EPOCH_WISE:
            progress_bar.close()

    def validate(self,
                 valid,
                 verbose=VERBOSE_BATCH_WISE
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

            self.models[X2Y].eval()
            self.models[Y2X].eval()
            # Iterate for whole valid-set.
            for idx, mini_batch in enumerate(progress_bar):
                x_0, y_0 = (mini_batch.src[0][:, 1:-1],  # Remove BOS and EOS
                            mini_batch.src[1] - 2
                            ), mini_batch.tgt[0][:, :-1]
                # |x_0| = (batch_size, length0)
                # |y_0| = (batch_size, length1)
                y_hat = self.models[X2Y](x_0, y_0)
                # |y_hat| = (batch_size, length1, output_size1)

                x_0, y_0_0, y_0_1, restore_indice = self._reordering(mini_batch.src[0][:, :-1],
                                                                mini_batch.tgt[0][:, 1:-1], # Remove BOS and EOS
                                                                mini_batch.tgt[1] - 2
                                                                )
                y_0 = (y_0_0, y_0_1)
                # |x_0| = (batch_size, length0)
                # |y_0| = (batch_size, length1)
                x_hat = self.models[Y2X](y_0, x_0).index_select(dim=0, index=restore_indice)
                # |x_hat| = (batch_size, length0, output_size0)

                x, y = mini_batch.src[0][:, 1:], mini_batch.tgt[0][:, 1:]
                losses = self._get_loss(x, y, x_hat, y_hat)

                word_count = int((mini_batch.src[1].detach().sum()) + 
                                 (mini_batch.tgt[1].detach().sum())
                                 )
                loss = float(losses[X2Y].detach() + losses[Y2X].detach())

                total_loss += float(loss)
                total_word_count += word_count
                avg_loss = total_loss / total_word_count

                sample_cnt += mini_batch.tgt[0].size(0)

                if verbose is VERBOSE_BATCH_WISE:
                    progress_bar.set_postfix_str('valid_loss=%.4e PPL=%.2f' % (avg_loss,
                                                                               exp(avg_loss)
                                                                               ))

                if idx >= len(progress_bar):
                    break

            self.models[X2Y].train()
            self.models[Y2X].train()

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.close()

            return avg_loss
