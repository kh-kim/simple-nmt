from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.utils as torch_utils

import utils

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

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
                     'current_lr': config.lr,
                     'config': config,
                     **kwargs
                     }

    def get_best_model(self):
        self.models[X2Y].load_state_dict(self.best['models'][X2Y])
        self.models[Y2X].load_state_dict(self.best['models'][Y2X])

        return self.models

    def save_training(self, fn):
        torch.save(self.best, fn)

    def _get_loss(self, x, y, x_hat, y_hat, x_lm=None, y_lm=None, lagrange=1e-2):
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

            losses[X2Y] += lagrange * ((lm_losses[Y2X] + losses[X2Y]) - (lm_losses[X2Y] + losses[Y2X].detach()))**2
            losses[Y2X] += lagrange * ((lm_losses[Y2X] + losses[X2Y].detach()) - (lm_losses[X2Y] + losses[Y2X]))**2

        return losses[X2Y].sum(), losses[Y2X].sum()

    def train_epoch(self,
                    train,
                    optimizers,
                    verbose=VERBOSE_BATCH_WISE
                    ):
        '''
        Train an epoch with given train iterator and optimizers.
        '''
        total_loss, total_word_count = 0, 0
        total_param_norm, total_grad_norm = 0, 0
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
            
            # You have to reset the gradients of all model parameters before to take another step in gradient descent.
            optimizers[X2Y].zero_grad()
            optimizers[Y2X].zero_grad()

            x_0, y_0 = (mini_batch.src[0][:, 1:-1], 
                        mini_batch.src[1] - 2
                        ), mini_batch.tgt[0][:, :-1]
            # |x_0| = (batch_size, length0)
            # |y_0| = (batch_size, length1)
            y_hat = self.models[X2Y](x_0, y_0)
            # |y_hat| = (batch_size, length1, output_size1)
            with torch.no_grad():
                y_lm = self.language_models[X2Y](y_0)
                # |y_lm| = |y_hat|

            x_0, y_0 = mini_batch.src[0][:, :-1], (mini_batch.tgt[0][:, 1:-1], 
                                                   mini_batch.tgt[1] - 2
                                                   )
            # |x_0| = (batch_size, length0)
            # |y_0| = (batch_size, length1)
            x_hat = self.models[Y2X](y_0, x_0)
            # |x_hat| = (batch_size, length0, output_size0)
            with torch.no_grad():
                x_lm = self.language_models[Y2X](x_0)
                # |x_lm| = |x_hat|

            x, y = mini_batch.src[0][:, 1:], mini_batch.tgt[0][:, 1:]
            losses = self._get_loss(x, y, x_hat, y_hat, x_lm, y_lm)
            
            losses[X2Y].div(y.size(0)).backward()
            losses[Y2X].div(x.size(0)).backward()

            total_loss += losses[X2Y].detach() + losses[Y2X].detach()
            total_word_count += (int(mini_batch.src[1].detach().sum()) + 
                                 int(mini_batch.tgt[1].detach().sum())
                                 )
            total_param_norm += (utils.get_parameter_norm(self.models[X2Y].parameters()).detach() + 
                                 utils.get_parameter_norm(self.models[Y2X].parameters()).detach()
                                 )
            total_grad_norm += (utils.get_grad_norm(self.models[X2Y].parameters()).detach() +
                                utils.get_grad_norm(self.models[Y2X].parameters()).detach()
                                )

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
            if sample_cnt >= len(train.dataset.examples):
                break

        if verbose is VERBOSE_BATCH_WISE:
            progress_bar.close()

        return avg_loss, avg_param_norm, avg_grad_norm

    def train(self, train, valid, verbose=VERBOSE_EPOCH_WISE):
        '''
        Train with given train and valid iterator until n_epochs.
        If early_stop is set,
        early stopping will be executed if the requirement is satisfied.
        '''
        current_lr = self.best['current_lr']
        best_loss = float('Inf')
        lowest_after = 0

        progress_bar = tqdm(range(self.best['epoch'], self.n_epochs),
                            desc='Training: ',
                            unit='epoch'
                            ) if verbose is VERBOSE_EPOCH_WISE else range(self.best['epoch'],
                                                                          self.n_epochs
                                                                          )

        for idx in progress_bar:  # Iterate from 1 to n_epochs
            if self.config.adam:
                optimizers = [optim.Adam(self.models[X2Y].parameters(), lr=current_lr),
                              optim.Adam(self.models[Y2X].parameters(), lr=current_lr)]
            else:
                optimizers = [optim.SGD(self.models[X2Y].parameters(), lr=current_lr),
                              optim.SGD(self.models[Y2X].parameters(), lr=current_lr)]

            if verbose > VERBOSE_EPOCH_WISE:
                print('epoch: %d/%d\tmin_valid_loss=%.4e' % (idx + 1,
                                                             self.n_epochs,
                                                             best_loss
                                                             ))
            avg_train_loss, avg_param_norm, avg_grad_norm = self.train_epoch(train,
                                                                             optimizers,
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

            if avg_valid_loss < best_loss:
                # Update if there is an improvement.
                best_loss = avg_valid_loss
                lowest_after = 0

                self.best['models'] = [self.models[X2Y].state_dict(),
                                       self.models[Y2X].state_dict()
                                       ]
                self.best['optim'] = optimizers
                self.best['epoch'] = idx + 1
                self.best['current_lr'] = current_lr

                # Set a filename for model of last epoch.
                # We need to put every information to filename, as much as possible.
                model_fn = self.config.model.split('.')
                model_fn = model_fn[:-1] + ['%02d' % (idx + 1),
                                            '%.2f-%.2f' % (avg_train_loss,
                                                           avg_train_loss.exp()
                                                           ),
                                            '%.2f-%.2f' % (avg_valid_loss,
                                                           avg_valid_loss.exp()
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
                x_0, y_0 = (mini_batch.src[0][:, 1:-1],
                            mini_batch.src[1] - 2
                            ), mini_batch.tgt[0][:, :-1]
                # |x_0| = (batch_size, length0)
                # |y_0| = (batch_size, length1)
                y_hat = self.models[X2Y](x_0, y_0)
                # |y_hat| = (batch_size, length1, output_size1)

                x_0, y_0 = mini_batch.src[0][:, :-1], (mini_batch.tgt[0][:, 1:-1],
                                                       mini_batch.tgt[1] - 2
                                                       )
                # |x_0| = (batch_size, length0)
                # |y_0| = (batch_size, length1)
                x_hat = self.models[Y2X](y_0, x_0)
                # |x_hat| = (batch_size, length0, output_size0)

                x, y = mini_batch.src[0][:, 1:], mini_batch.tgt[0][:, 1:]
                losses = self._get_loss(x, y, x_hat, y_hat)

                total_loss += losses[X2Y].detach() + losses[Y2X].detach()
                total_word_count += (int(mini_batch.src[1].detach().sum()) + 
                                     int(mini_batch.tgt[1].detach().sum())
                                     )
                avg_loss = total_loss / total_word_count

                sample_cnt += mini_batch.tgt[0].size(0)

                if verbose is VERBOSE_BATCH_WISE:
                    progress_bar.set_postfix_str('valid_loss=%.4e PPL=%.2f' % (avg_loss,
                                                                               avg_loss.exp()
                                                                               ))

                if sample_cnt >= len(valid.dataset.examples):
                    break
            self.models[X2Y].train()
            self.models[Y2X].train()

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.close()

            return avg_loss