from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

import numpy as np

import torch
from torch import optim
from torch.nn import functional as F
import torch.nn.utils as torch_utils

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

import simple_nmt.data_loader as data_loader
from simple_nmt.trainer import MaximumLikelihoodEstimationEngine
from simple_nmt.utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MinimumRiskTrainingEngine(MaximumLikelihoodEstimationEngine):

    @staticmethod
    def _get_reward(y_hat, y, n_gram=6, method='gleu'):
        # This method gets the reward based on the sampling result and reference sentence.
        # For now, we uses GLEU in NLTK, but you can used your own well-defined reward function.
        # In addition, GLEU is variation of BLEU, and it is more fit to reinforcement learning.
        sf = SmoothingFunction()
        score_func = {
            'gleu':  lambda ref, hyp: sentence_gleu([ref], hyp, max_len=n_gram),
            'bleu1': lambda ref, hyp: sentence_bleu([ref], hyp,
                                                    weights=[1./n_gram] * n_gram,
                                                    smoothing_function=sf.method1),
            'bleu2': lambda ref, hyp: sentence_bleu([ref], hyp,
                                                    weights=[1./n_gram] * n_gram,
                                                    smoothing_function=sf.method2),
            'bleu4': lambda ref, hyp: sentence_bleu([ref], hyp,
                                                    weights=[1./n_gram] * n_gram,
                                                    smoothing_function=sf.method4),
        }[method]

        # Since we don't calculate reward score exactly as same as multi-bleu.perl,
        # (especialy we do have different tokenization,) I recommend to set n_gram to 6.

        # |y| = (batch_size, length1)
        # |y_hat| = (batch_size, length2)

        with torch.no_grad():
            scores = []

            for b in range(y.size(0)):
                ref, hyp = [], []
                for t in range(y.size(-1)):
                    ref += [str(int(y[b, t]))]
                    if y[b, t] == data_loader.EOS:
                        break

                for t in range(y_hat.size(-1)):
                    hyp += [str(int(y_hat[b, t]))]
                    if y_hat[b, t] == data_loader.EOS:
                        break
                # Below lines are slower than naive for loops in above.
                # ref = y[b].masked_select(y[b] != data_loader.PAD).tolist()
                # hyp = y_hat[b].masked_select(y_hat[b] != data_loader.PAD).tolist()

                scores += [score_func(ref, hyp) * 100.]
            scores = torch.FloatTensor(scores).to(y.device)
            # |scores| = (batch_size)

            return scores


    @staticmethod
    def _get_loss(y_hat, indice, reward=1):
        # |indice| = (batch_size, length)
        # |y_hat| = (batch_size, length, output_size)
        # |reward| = (batch_size,)
        batch_size = indice.size(0)
        output_size = y_hat.size(-1)

        '''
        # Memory inefficient but more readable version
        mask = indice == data_loader.PAD
        # |mask| = (batch_size, length)
        indice = F.one_hot(indice, num_classes=output_size).float()
        # |indice| = (batch_size, length, output_size)
        log_prob = (y_hat * indice).sum(dim=-1)
        # |log_prob| = (batch_size, length)
        log_prob.masked_fill_(mask, 0)
        log_prob = log_prob.sum(dim=-1)
        # |log_prob| = (batch_size, )
        '''

        # Memory efficient version
        log_prob = -F.nll_loss(
            y_hat.view(-1, output_size),
            indice.view(-1),
            ignore_index=data_loader.PAD,
            reduction='none'
        ).view(batch_size, -1).sum(dim=-1)

        loss = (log_prob * -reward).sum()
        # Following two equations are eventually same.
        # \theta = \theta - risk * \nabla_\theta \log{P}
        # \theta = \theta - -reward * \nabla_\theta \log{P}
        # where risk = -reward.

        return loss

    @staticmethod
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()
        if engine.state.iteration % engine.config.iteration_per_update == 1 or \
            engine.config.iteration_per_update == 1:
            if engine.state.iteration > 1:
                engine.optimizer.zero_grad()

        device = next(engine.model.parameters()).device
        mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
        mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

        # Raw target variable has both BOS and EOS token.
        # The output of sequence-to-sequence does not have BOS token.
        # Thus, remove BOS token for reference.
        x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
        # |x| = (batch_size, length)
        # |y| = (batch_size, length)

        # Take sampling process because set False for is_greedy.
        y_hat, indice = engine.model.search(
            x,
            is_greedy=False,
            max_length=engine.config.max_length
        )

        with torch.no_grad():
            # Based on the result of sampling, get reward.
            actor_reward = MinimumRiskTrainingEngine._get_reward(
                indice,
                y,
                n_gram=engine.config.rl_n_gram,
                method=engine.config.rl_reward,
            )
            # |y_hat| = (batch_size, length, output_size)
            # |indice| = (batch_size, length)
            # |actor_reward| = (batch_size)

            # Take samples as many as n_samples, and get average rewards for them.
            # I figured out that n_samples = 1 would be enough.
            baseline = []

            for _ in range(engine.config.rl_n_samples):
                _, sampled_indice = engine.model.search(
                    x,
                    is_greedy=False,
                    max_length=engine.config.max_length,
                )
                baseline += [
                    MinimumRiskTrainingEngine._get_reward(
                        sampled_indice,
                        y,
                        n_gram=engine.config.rl_n_gram,
                        method=engine.config.rl_reward,
                    )
                ]

            baseline = torch.stack(baseline).mean(dim=0)
            # |baseline| = (n_samples, batch_size) --> (batch_size)

            # Now, we have relatively expected cumulative reward.
            # Which score can be drawn from actor_reward subtracted by baseline.
            reward = actor_reward - baseline
            # |reward| = (batch_size)

        # calculate gradients with back-propagation
        loss = MinimumRiskTrainingEngine._get_loss(
            y_hat,
            indice,
            reward=reward
        )
        backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)
        backward_target.backward()

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        if engine.state.iteration % engine.config.iteration_per_update == 0 and \
            engine.state.iteration > 0:
            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad_norm,
            )
            # Take a step of gradient descent.
            engine.optimizer.step()

        return {
            'actor': float(actor_reward.mean()),
            'baseline': float(baseline.mean()),
            'reward': float(reward.mean()),
            '|param|': p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
            '|g_param|': g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device
            mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            # feed-forward
            y_hat, indice = engine.model.search(
                x,
                is_greedy=True,
                max_length=engine.config.max_length,
            )
            # |y_hat| = (batch_size, length, output_size)
            # |indice| = (batch_size, length)
            reward = MinimumRiskTrainingEngine._get_reward(
                indice,
                y,
                n_gram=engine.config.rl_n_gram,
                method=engine.config.rl_reward,
            )

        return {
            'BLEU': float(reward.mean()),
        }

    @staticmethod
    def attach(
        train_engine,
        validation_engine,
        training_metric_names = ['actor', 'baseline', 'reward', '|param|', '|g_param|'],
        validation_metric_names = ['BLEU', ],
        verbose=VERBOSE_BATCH_WISE
    ):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_reward = engine.state.metrics['actor']

                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} BLEU={:.2f}'.format(
                    engine.state.epoch,
                    avg_p_norm,
                    avg_g_norm,
                    avg_reward,
                ))

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_bleu = engine.state.metrics['BLEU']
                print('Validation - BLEU={:.2f} best_BLEU={:.2f}'.format(
                    avg_bleu,
                    -engine.best_loss,
                ))

    @staticmethod
    def resume_training(engine, resume_epoch):
        resume_epoch = max(1, resume_epoch - engine.config.n_epochs)
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)

    @staticmethod
    def check_best(engine):
        loss = -float(engine.state.metrics['BLEU'])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_bleu = train_engine.state.metrics['actor']
        avg_valid_bleu = engine.state.metrics['BLEU']

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split('.')

        model_fn = model_fn[:-1] + ['mrt',
                                    '%02d' % train_engine.state.epoch,
                                    '%.2f-%.2f' % (avg_train_bleu,
                                                   avg_valid_bleu),
                                    ] + [model_fn[-1]]

        model_fn = '.'.join(model_fn)

        # Unlike other tasks, we need to save current model, not best model.
        torch.save(
            {
                'model': engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
            }, model_fn
        )
