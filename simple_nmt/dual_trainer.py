import numpy as np
import torch

from torch import optim
import torch.nn.utils as torch_utils
from ignite.engine import Engine, Events

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

X2Y, Y2X = 0, 1


class DualSupervisedTrainer():

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _reordering(x, y, l):
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

        return x_, (y_, l_), restore_indice

    @staticmethod
    def _get_loss(x, y, x_hat, y_hat, crits, x_lm=None, y_lm=None, lagrange=1e-3):
        # |x| = (batch_size, length0)
        # |y| = (batch_size, length1)
        # |x_hat| = (batch_size, length0, output_size0)
        # |y_hat| = (batch_size, length1, output_size1)
        # |x_lm| = |x_hat|
        # |y_lm| = |y_hat|

        loss_x2y = crits[X2Y](
            y_hat.contiguous().view(-1, y_hat.size(-1)),
            y.contiguous().view(-1),
        )
        loss_y2x = crits[Y2X](
            x_hat.contiguous().view(-1, x_hat.size(-1)),
            x.contiguous().view(-1),
        )
        # |loss_x2y| = (batch_size * m)
        # |loss_y2x| = (batch_size * n)

        loss_x2y = loss_x2y.view(y.size(0), -1).sum(dim=-1)
        loss_y2x = loss_y2x.view(x.size(0), -1).sum(dim=-1)
        # |loss_x2y| = |loss_y2x| = (batch_size, )

        if x_lm is not None and y_lm is not None:
            lm_loss_x2y = crits[X2Y](
                y_lm.contiguous().view(-1, y_lm.size(-1)),
                y.contiguous().view(-1),
            )
            lm_loss_y2x = crits[Y2X](
                x_lm.contiguous().view(-1, x_lm.size(-1)),
                x.contiguous().view(-1),
            )
            # |lm_loss_x2y| = (batch_size * m)
            # |lm_loss_y2x| = (batch_size * n)

            lm_loss_x2y = lm_loss_x2y.view(y.size(0), -1).sum(dim=-1)
            lm_loss_y2x = lm_loss_y2x.view(x.size(0), -1).sum(dim=-1)
            # |lm_loss_x2y| = (batch_size, )
            # |lm_loss_y2x| = (batch_size, )

            # Just for logging: both losses are detached.
            dual_loss = lagrange * ((-lm_loss_y2x + -loss_x2y.detach()) - (-lm_loss_x2y + -loss_y2x.detach()))**2

            # Note that 'detach()' is used to prevent unnecessary back-propagation.
            loss_x2y += lagrange * ((-lm_loss_y2x + -loss_x2y) - (-lm_loss_x2y + -loss_y2x.detach()))**2
            loss_y2x += lagrange * ((-lm_loss_y2x + -loss_x2y.detach()) - (-lm_loss_x2y + -loss_y2x))**2
        else:
            dual_loss = None

        return (
            loss_x2y.sum(),
            loss_y2x.sum(),
            dual_loss.sum() if dual_loss is not None else .0,
        )

    @staticmethod
    def step(engine, mini_batch):
        from utils import get_grad_norm, get_parameter_norm

        for language_model, model, optimizer in zip(engine.language_models, engine.models, engine.optimizers):
            language_model.eval()
            model.train()
            optimizer.zero_grad()

        # X2Y
        x, y = (mini_batch.src[0][:, 1:-1], mini_batch.src[1] - 2), mini_batch.tgt[0][:, :-1]
        # |x| = (batch_size, n)
        # |y| = (batch_size, m)
        y_hat = engine.models[X2Y](x, y)
        # |y_hat| = (batch_size, m, y_vocab_size)
        with torch.no_grad():
            p_hat_y = engine.language_models[X2Y](y)
            # |p_hat_y| = |y_hat|

        #Y2X
        # Since encoder in seq2seq takes packed_sequence instance,
        # we need to re-sort if we use reversed src and tgt.
        x, y, restore_indice = DualSupervisedTrainer._reordering(
            mini_batch.src[0][:, :-1],
            mini_batch.tgt[0][:, 1:-1],
            mini_batch.tgt[1] - 2,
        )
        # |x| = (batch_size, n)
        # |y| = (batch_size, m)
        x_hat = engine.models[Y2X](y, x).index_select(dim=0, index=restore_indice)
        # |x_hat| = (batch_size, n, x_vocab_size)

        with torch.no_grad():
            p_hat_x = engine.language_models[Y2X](x).index_select(dim=0, index=restore_indice)
            # |p_hat_x| = |x_hat|

        x, y = mini_batch.src[0][:, 1:], mini_batch.tgt[0][:, 1:]
        loss_x2y, loss_y2x, dual_loss = DualSupervisedTrainer._get_loss(
            x, y,
            x_hat, y_hat,
            engine.crits,
            p_hat_x, p_hat_y,
            # According to the paper, DSL should be warm-started.
            # Thus, we turn-off the regularization at the beginning.
            lagrange=engine.config.dsl_lambda if engine.state.epoch >= engine.config.n_epochs else .0
        )

        loss_x2y.div(y.size(0)).backward()
        loss_y2x.div(x.size(0)).backward()

        p_norm = float(get_parameter_norm(list(engine.models[X2Y].parameters()) + 
                                          list(engine.models[Y2X].parameters())))
        g_norm = float(get_grad_norm(list(engine.models[X2Y].parameters()) +
                                     list(engine.models[Y2X].parameters())))

        for model, optimizer in zip(engine.models, engine.optimizers):
            torch_utils.clip_grad_norm_(
                model.parameters(),
                engine.config.max_grad_norm,
            )
            # Take a step of gradient descent.
            optimizer.step()

        return (
            float(loss_x2y / mini_batch.src[1].sum()),
            float(loss_y2x / mini_batch.tgt[1].sum()),
            float(dual_loss / x.size(0)),
            p_norm,
            g_norm,
        )

    @staticmethod
    def validate(engine, mini_batch):
        for model in engine.models:
            model.eval()

        with torch.no_grad():
            # X2Y
            x, y = (mini_batch.src[0][:, 1:-1], mini_batch.src[1] - 2), mini_batch.tgt[0][:, :-1]
            # |x| = (batch_size, n)
            # |y| = (batch_size  m)
            y_hat = engine.models[X2Y](x, y)
            # |y_hat| = (batch_size, m, y_vocab_size)

            # Y2X
            x, y, restore_indice = DualSupervisedTrainer._reordering(
                mini_batch.src[0][:, :-1],
                mini_batch.tgt[0][:, 1:-1],
                mini_batch.tgt[1] - 2,
            )
            x_hat = engine.models[Y2X](y, x).index_select(dim=0, index=restore_indice)
            # |x_hat| = (batch_size, n, x_vocab_size)

            x, y = mini_batch.src[0][:, 1:], mini_batch.tgt[0][:, 1:]
            loss_x2y = engine.crits[X2Y](
                y_hat.contiguous().view(-1, y_hat.size(-1)),
                y.contiguous().view(-1)
            ).sum()
            loss_y2x = engine.crits[Y2X](
                x_hat.contiguous().view(-1, x_hat.size(-1)),
                x.contiguous().view(-1)
            ).sum()

        return float(loss_x2y / mini_batch.src[1].sum()), float(loss_y2x / mini_batch.tgt[1].sum())

    @staticmethod
    def attach(trainer, evaluator, verbose=VERBOSE_BATCH_WISE):
        from ignite.engine import Events
        from ignite.metrics import RunningAverage
        from ignite.contrib.handlers.tqdm_logger import ProgressBar

        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'x2y')
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'y2x')
        RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'reg')
        RunningAverage(output_transform=lambda x: x[3]).attach(trainer, '|param|')
        RunningAverage(output_transform=lambda x: x[4]).attach(trainer, '|g_param|')

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(trainer, ['|param|', '|g_param|', 'x2y', 'y2x', 'reg'])

        if verbose >= VERBOSE_EPOCH_WISE:
            @trainer.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_x2y = engine.state.metrics['x2y']
                avg_y2x = engine.state.metrics['y2x']
                avg_reg = engine.state.metrics['reg']

                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss_x2y={:.4e} ppl_x2y={:.2f} loss_x2y={:.4e} ppl_x2y={:.2f} dual_loss={:.4e}'.format(
                    engine.state.epoch,
                    avg_p_norm,
                    avg_g_norm,
                    avg_x2y, np.exp(avg_x2y),
                    avg_y2x, np.exp(avg_y2x),
                    avg_reg,
                ))

        RunningAverage(output_transform=lambda x: x[0]).attach(evaluator, 'x2y')
        RunningAverage(output_transform=lambda x: x[1]).attach(evaluator, 'y2x')

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(evaluator, ['x2y', 'y2x'])

        if verbose >= VERBOSE_EPOCH_WISE:
            @evaluator.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_x2y = engine.state.metrics['x2y']
                avg_y2x = engine.state.metrics['y2x']

                print('Validation X2Y - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_x2y,
                    np.exp(avg_x2y),
                    engine.best_x2y,
                    np.exp(engine.best_x2y),
                ))
                print('Validation Y2X - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_y2x,
                    np.exp(avg_y2x),
                    engine.best_y2x,
                    np.exp(engine.best_y2x),
                ))

    @staticmethod
    def check_best(engine):
        from copy import deepcopy

        x2y = float(engine.state.metrics['x2y'])
        if x2y <= engine.best_x2y:
            engine.best_x2y = x2y
        y2x = float(engine.state.metrics['y2x'])
        if y2x <= engine.best_y2x:
            engine.best_y2x = y2x

    @staticmethod
    def save_model(engine, train_engine, config, vocabs):
        avg_train_x2y = train_engine.state.metrics['x2y']
        avg_train_y2x = train_engine.state.metrics['y2x']
        avg_valid_x2y = engine.state.metrics['x2y']
        avg_valid_y2x = engine.state.metrics['y2x']

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split('.')
        
        model_fn = model_fn[:-1] + ['%02d' % ((config.init_epoch - 1) + train_engine.state.epoch),
                                    '%.2f-%.2f' % (avg_train_x2y,
                                                   np.exp(avg_train_x2y)
                                                   ),
                                    '%.2f-%.2f' % (avg_train_y2x,
                                                   np.exp(avg_train_y2x)
                                                   ),
                                    '%.2f-%.2f' % (avg_valid_x2y,
                                                   np.exp(avg_valid_x2y)
                                                   ),
                                    '%.2f-%.2f' % (avg_valid_y2x,
                                                   np.exp(avg_valid_y2x)
                                                   ),
                                    ] + [model_fn[-1]]

        model_fn = '.'.join(model_fn)

        torch.save(
            {
                'model': [
                    train_engine.models[0].state_dict(),
                    train_engine.models[1].state_dict(),
                    train_engine.language_models[0].state_dict(),
                    train_engine.language_models[1].state_dict(),
                ],
                'opt': [
                    train_engine.optimizers[0].state_dict(),
                    train_engine.optimizers[1].state_dict(),
                ],
                'config': config,
                'src_vocab': vocabs[0],
                'tgt_vocab': vocabs[1],
            }, model_fn
        )

    def train(
        self,
        models, language_models,
        crits, optimizers,
        train_loader, valid_loader,
        vocabs,
        n_epochs,
        lr_schedulers=None
    ):
        trainer = Engine(self.step)
        trainer.config = self.config
        trainer.models, trainer.crits = models, crits
        trainer.optimizers, trainer.lr_schedulers = optimizers, lr_schedulers
        trainer.language_models = language_models
        trainer.epoch_idx = 0

        evaluator = Engine(self.validate)
        evaluator.config = self.config
        evaluator.models, evaluator.crits = models, crits
        evaluator.best_x2y, evaluator.best_y2x = np.inf, np.inf

        self.attach(trainer, evaluator, verbose=self.config.verbose)

        def run_validation(engine, evaluator, valid_loader):
            evaluator.run(valid_loader, max_epochs=1)

            if engine.lr_schedulers is not None:
                for s in engine.lr_schedulers:
                    s.step()

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
            vocabs,
        )

        trainer.run(train_loader, max_epochs=n_epochs)

        return models
