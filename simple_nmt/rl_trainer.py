from tqdm import tqdm
# from nltk.translate.bleu_score import sentence_bleu as score_func
from nltk.translate.gleu_score import sentence_gleu as score_func
# from utils import score_sentence as score_func

import torch
import torch.nn.utils as torch_utils

import utils
import data_loader

from simple_nmt.trainer import Trainer

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MinimumRiskTrainer(Trainer):

    def __init__(self, model, crit, config, **kwargs):
        super().__init__(model=model, crit=crit, config=config, **kwargs)

        self.n_epochs = config.n_epochs + config.rl_n_epochs
        self.lower_is_better = False
        self.best = {'epoch': config.n_epochs,
                     'current_lr': config.rl_lr,
                     'config': config,
                     **kwargs
                     }

    def _get_reward(self, y_hat, y, n_gram=6):
        # This method gets the reward based on the sampling result and reference sentence.
        # For now, we uses GLEU in NLTK, but you can used your own well-defined reward function.
        # In addition, GLEU is variation of BLEU, and it is more fit to reinforcement learning.

        # Since we don't calculate reward score exactly as same as multi-bleu.perl,
        # (especialy we do have different tokenization,) I recommend to set n_gram to 6.

        # |y| = (batch_size, length1)
        # |y_hat| = (batch_size, length2)

        scores = []

        # Actually, below is really far from parallized operations.
        # Thus, it may cause slow training.
        for b in range(y.size(0)):
            ref = []
            hyp = []
            for t in range(y.size(1)):
                ref += [str(int(y[b, t]))]
                if y[b, t] == data_loader.EOS:
                    break

            for t in range(y_hat.size(1)):
                hyp += [str(int(y_hat[b, t]))]
                if y_hat[b, t] == data_loader.EOS:
                    break

            # for nltk.bleu & nltk.gleu
            scores += [score_func([ref], hyp, max_len=n_gram) * 100.]

            # for utils.score_sentence
            # scores += [score_func(ref, hyp, 4, smooth = 1)[-1] * 100.]
        scores = torch.FloatTensor(scores).to(y.device)
        # |scores| = (batch_size)

        return scores

    def _get_gradient(self, y_hat, y, crit=None, reward=1):
        # |y| = (batch_size, length)
        # |y_hat| = (batch_size, length, output_size)
        # |reward| = (batch_size)
        crit = self.crit if crit is None else crit

        # Before we get the gradient, multiply -reward for each sample and each time-step.
        y_hat = y_hat * -reward.view(-1, 1, 1).expand(*y_hat.size())

        # Again, multiply -1 because criterion is NLLLoss.
        log_prob = -crit(y_hat.contiguous().view(-1, y_hat.size(-1)),
                         y.contiguous().view(-1)
                         )
        log_prob.div(y.size(0)).backward()

        return log_prob

    def train_epoch(self,
                    train,
                    optimizer,
                    max_grad_norm=5,
                    verbose=VERBOSE_SILENT
                    ):
        '''
        Train an epoch with given train iterator and optimizer.
        '''
        total_reward, total_actor_reward = 0, 0
        total_grad_norm = 0
        avg_reward, avg_actor_reward = 0, 0
        avg_param_norm, avg_grad_norm = 0, 0
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

            # Take sampling process because set False for is_greedy.
            y_hat, indice = self.model.search(x,
                                              is_greedy=False,
                                              max_length=self.config.max_length
                                              )
            # Based on the result of sampling, get reward.
            actor_reward = self._get_reward(indice,
                                            y,
                                            n_gram=self.config.rl_n_gram
                                            )
            # |y_hat| = (batch_size, length, output_size)
            # |indice| = (batch_size, length)
            # |actor_reward| = (batch_size)

            # Take samples as many as n_samples, and get average rewards for them.
            # I figured out that n_samples = 1 would be enough.
            baseline = []
            with torch.no_grad():
                for i in range(self.config.n_samples):
                    _, sampled_indice = self.model.search(x,
                                                          is_greedy=False,
                                                          max_length=self.config.max_length
                                                          )
                    baseline += [self._get_reward(sampled_indice,
                                                  y,
                                                  n_gram=self.config.rl_n_gram
                                                  )]
                baseline = torch.stack(baseline).sum(dim=0).div(self.config.n_samples)
                # |baseline| = (n_samples, batch_size) --> (batch_size)

            # Now, we have relatively expected cumulative reward.
            # Which score can be drawn from actor_reward subtracted by baseline.
            final_reward = actor_reward - baseline
            # |final_reward| = (batch_size)

            # calcuate gradients with back-propagation
            self._get_gradient(y_hat, indice, reward=final_reward)
            
            # Simple math to show stats.
            total_reward += float(final_reward.sum())
            total_actor_reward += float(actor_reward.sum())
            sample_cnt += int(actor_reward.size(0))
            param_norm = float(utils.get_parameter_norm(self.model.parameters()))
            total_grad_norm += float(utils.get_grad_norm(self.model.parameters()))

            avg_reward = total_reward / sample_cnt
            avg_actor_reward = total_actor_reward / sample_cnt
            avg_grad_norm = total_grad_norm / (idx + 1)

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.set_postfix_str('|g_param|=%.2f rwd=%4.2f avg_frwd=%.2e BLEU=%.4f' % (avg_grad_norm,
                                                                                                 float(actor_reward.sum().div(y.size(0))),
                                                                                                 avg_reward,
                                                                                                 avg_actor_reward
                                                                                                 ))

            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(self.model.parameters(),
                                        self.config.max_grad_norm
                                        )
            # Take a step of gradient descent.
            optimizer.step()


            if idx >= len(progress_bar) * self.config.train_ratio_per_epoch:
                break

        if verbose is VERBOSE_BATCH_WISE:
            progress_bar.close()

        return avg_actor_reward, param_norm, avg_grad_norm

    def validate(self,
                 valid,
                 crit=None,
                 verbose=VERBOSE_SILENT
                 ):
        '''
        Validate a model with given valid iterator.
        '''
        # We don't need to back-prop for these operations.
        with torch.no_grad():
            total_reward, sample_cnt = 0, 0

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

                # feed-forward
                y_hat, indice = self.model.search(x,
                                                  is_greedy=True,
                                                  max_length=self.config.max_length
                                                  )
                # |y_hat| = (batch_size, length, output_size)
                # |indice| = (batch_size, length)
                reward = self._get_reward(indice,
                                          y,
                                          n_gram=self.config.rl_n_gram
                                          )

                total_reward += float(reward.sum())
                sample_cnt += int(mini_batch.tgt[0].size(0))
                
                avg_reward = total_reward / sample_cnt

                if verbose is VERBOSE_BATCH_WISE:
                    progress_bar.set_postfix_str('valid_BLEU=%.4f' % avg_reward)

                if idx >= len(progress_bar):
                    break

            self.model.train()

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.close()

            return avg_reward
