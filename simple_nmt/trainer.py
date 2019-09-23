import numpy as np
import torch

from torch import optim
import torch.nn.utils as torch_utils
from ignite.engine import Engine, Events

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MaximumLikelihoodEstimationTrainer():

    def __init__(self, config):
        self.config = config

    @staticmethod
    def step(engine, mini_batch):
        from utils import get_grad_norm, get_parameter_norm

        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()        
        engine.optimizer.zero_grad()

        # Raw target variable has both BOS and EOS token. 
        # The output of sequence-to-sequence does not have BOS token. 
        # Thus, remove BOS token for reference.
        x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
        # |x| = (batch_size, length)
        # |y| = (batch_size, length)

        # Take feed-forward
        # Similar as before, the input of decoder does not have EOS token.
        # Thus, remove EOS token for decoder input.
        y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
        # |y_hat| = (batch_size, length, output_size)

        loss = engine.crit(y_hat.contiguous().view(-1, y_hat.size(-1)),
                           y.contiguous().view(-1)
                           )
        loss.div(y.size(0)).backward()
        word_count = int(mini_batch.tgt[1].sum())

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # In orther to avoid gradient exploding, we apply gradient clipping.
        torch_utils.clip_grad_norm_(
            engine.model.parameters(),
            engine.config.max_grad_norm,
        )
        # Take a step of gradient descent.
        engine.optimizer.step()

        if engine.config.use_noam_decay and engine.lr_scheduler is not None:
            engine.lr_scheduler.step()

        return float(loss / word_count), p_norm, g_norm

    @staticmethod
    def validate(engine, mini_batch):
        from utils import get_grad_norm, get_parameter_norm

        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)
            y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
            # |y_hat| = (batch_size, n_classes)
            loss = engine.crit(
                y_hat.contiguous().view(-1, y_hat.size(-1)),
                y.contiguous().view(-1),
            )
            word_count = int(mini_batch.tgt[1].sum())

        return float(loss / word_count)

    @staticmethod
    def attach(trainer, evaluator, verbose=VERBOSE_BATCH_WISE):
        from ignite.engine import Events
        from ignite.metrics import RunningAverage
        from ignite.contrib.handlers.tqdm_logger import ProgressBar

        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, '|param|')
        RunningAverage(output_transform=lambda x: x[2]).attach(trainer, '|g_param|')

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(trainer, ['|param|', '|g_param|', 'loss'])

        if verbose >= VERBOSE_EPOCH_WISE:
            @trainer.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_loss = engine.state.metrics['loss']

                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} ppl={:.2f}'.format(
                    engine.state.epoch,
                    avg_p_norm,
                    avg_g_norm,
                    avg_loss,
                    np.exp(avg_loss),
                ))

        RunningAverage(output_transform=lambda x: x).attach(evaluator, 'loss')

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(evaluator, ['loss'])

        if verbose >= VERBOSE_EPOCH_WISE:
            @evaluator.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_loss = engine.state.metrics['loss']
                print('Validation - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_loss,
                    np.exp(avg_loss),
                    engine.best_loss,
                    np.exp(engine.best_loss),
                ))

    @staticmethod
    def check_best(engine):
        from copy import deepcopy

        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_loss = train_engine.state.metrics['loss']
        avg_valid_loss = engine.state.metrics['loss']

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split('.')
        
        model_fn = model_fn[:-1] + ['%02d' % ((config.init_epoch - 1) + train_engine.state.epoch),
                                    '%.2f-%.2f' % (avg_train_loss,
                                                   np.exp(avg_train_loss)
                                                   ),
                                    '%.2f-%.2f' % (avg_valid_loss,
                                                   np.exp(avg_valid_loss)
                                                   )
                                    ] + [model_fn[-1]]

        model_fn = '.'.join(model_fn)

        torch.save(
            {
                'model': engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
            }, model_fn
        )

    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader,
        src_vocab, tgt_vocab,
        n_epochs,
        lr_scheduler=None
    ):
        trainer = Engine(self.step)
        trainer.config = self.config
        trainer.model, trainer.crit = model, crit
        trainer.optimizer, trainer.lr_scheduler = optimizer, lr_scheduler
        trainer.epoch_idx = 0

        evaluator = Engine(self.validate)
        evaluator.config = self.config
        evaluator.model, evaluator.crit = model, crit
        evaluator.best_loss = np.inf

        self.attach(trainer, evaluator, verbose=self.config.verbose)

        def run_validation(engine, evaluator, valid_loader):
            evaluator.run(valid_loader, max_epochs=1)

            if engine.lr_scheduler is not None and not engine.config.use_noam_decay:
                engine.lr_scheduler.step()

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, run_validation, evaluator, valid_loader
        )
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, self.check_best
        )
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.save_model,
            trainer,
            self.config,
            src_vocab,
            tgt_vocab,
        )

        trainer.run(train_loader, max_epochs=n_epochs)

        return model
