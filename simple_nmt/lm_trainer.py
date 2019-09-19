import numpy as np
import torch

from torch import optim
import torch.nn.utils as torch_utils
from ignite.engine import Engine, Events

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


from simple_nmt.trainer import MaximumLikelihoodEstimationTrainer


class LanguageModelTrainer(MaximumLikelihoodEstimationTrainer):

    @staticmethod
    def step(engine, mini_batch):
        from utils import get_grad_norm, get_parameter_norm

        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()        
        engine.optimizer.zero_grad()

        # if 'is_src_target' is true, the trainer would train language model for source language.
        # For dsl case, both x and y has BOS and EOS tokens.
        # Thus, we need to remove BOS and EOS before the training.
        x = mini_batch.src[0][:, :-1] if engine.is_src_target else mini_batch.tgt[0][:, :-1]
        y = mini_batch.src[0][:, 1:] if engine.is_src_target else mini_batch.tgt[0][:, 1:]
        # |x| = |y| = (batch_size, length)

        y_hat = engine.model(x)
        # |y_hat| = (batch_size, length, output_size)

        loss = engine.crit(y_hat.contiguous().view(-1, y_hat.size(-1)),
                           y.contiguous().view(-1),
                           ).sum()
        loss.div(y.size(0)).backward()
        word_count = int(mini_batch.src[1].sum()) if engine.is_src_target else int(mini_batch.tgt[1].sum())

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # In orther to avoid gradient exploding, we apply gradient clipping.
        torch_utils.clip_grad_norm_(
            engine.model.parameters(),
            engine.config.max_grad_norm,
        )
        # Take a step of gradient descent.
        engine.optimizer.step()

        return float(loss / word_count), p_norm, g_norm

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x = mini_batch.src[0][:, :-1] if engine.is_src_target else mini_batch.tgt[0][:, :-1]
            y = mini_batch.src[0][:, 1:] if engine.is_src_target else mini_batch.tgt[0][:, 1:]
            # |x| = |y| = (batch_size, length)

            y_hat = engine.model(x)
            # |y_hat| = (batch_size, length, output_size)

            loss = engine.crit(y_hat.contiguous().view(-1, y_hat.size(-1)),
                               y.contiguous().view(-1),
                               ).sum()
            word_count = int(mini_batch.src[1].sum()) if engine.is_src_target else int(mini_batch.tgt[1].sum())

        return float(loss / word_count)

    @staticmethod
    def check_best(engine):
        from copy import deepcopy

        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        pass

    def train(
        self,
        model,
        crit, optimizer,
        train_loader, valid_loader,
        src_vocab, tgt_vocab,
        n_epochs,
        lr_scheduler=None
    ):
        if src_vocab is not None and tgt_vocab is not None:
            raise NotImplementedError('You should assign None one of vocab to designate target language.')
        if src_vocab is None:
            is_src_target = False
        elif tgt_vocab is None:
            is_src_target = True
        else:
            raise NotImplementedError('You cannot assign None both vocab.')

        trainer = Engine(self.step)
        trainer.config = self.config
        trainer.model, trainer.crit = model, crit
        trainer.optimizer, trainer.lr_scheduler = optimizer, lr_scheduler
        trainer.epoch_idx = 0
        trainer.is_src_target = is_src_target

        evaluator = Engine(self.validate)
        evaluator.config = self.config
        evaluator.model, evaluator.crit = model, crit
        evaluator.best_loss = np.inf
        evaluator.is_src_target = is_src_target

        self.attach(trainer, evaluator, verbose=self.config.verbose)

        def run_validation(engine, evaluator, valid_loader):
            evaluator.run(valid_loader, max_epochs=1)

            if engine.lr_scheduler is not None:
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

        if n_epochs > 0:
            model.load_state_dict(evaluator.best_model)

        return model
